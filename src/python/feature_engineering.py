"""
E-Commerce Churn Prediction - Feature Engineering Pipeline
Purpose: Transform transaction data into customer-level features for ML modeling
Input: data/interim/cleaned_retail_data.csv (transaction-level)
Output: data/processed/customers_features.csv (customer-level)
Version: 1.0 (Simple feature set)
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from python.utils import (
    log_info, log_warn, log_error,
    validate_date_column, validate_no_missing_values,
    check_data_leakage, generate_feature_report, split_by_time_window
)


# --- Load Configuration ---

config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract time windows from config
OBS_START = config['features']['observation_start']
OBS_END = config['features']['observation_end']
OUTCOME_START = config['features']['outcome_start']
OUTCOME_END = config['features']['outcome_end']

# File paths
CLEANED_DATA_PATH = Path(config['paths']['interim_data']) / "cleaned_retail_data.csv"
OUTPUT_PATH = Path(config['paths']['processed_data']) / "customers_features.csv"

log_info("Configuration loaded successfully")


# --- Step 1: Load Cleaned Transaction Data ---
# Read CSV with proper data types to prevent parsing issues

log_info("Loading cleaned transaction data...")

df = pd.read_csv(
    CLEANED_DATA_PATH,
    parse_dates=['invoice_date', 'invoice_date_only'],
    dtype={
        'invoice': str,
        'stock_code': str,
        'customer_id': int,
        'quantity': int,
        'price': float,
        'country': str
    }
)

log_info(f"Data loaded: {len(df):,} transactions")

# Validate critical columns
validate_date_column(df, 'invoice_date')
validate_no_missing_values(df, ['customer_id', 'invoice_date', 'quantity', 'price'])


# --- Step 2: Split into Observation and Outcome Windows ---
# Observation window: Calculate features (18 months)
# Outcome window: Determine churn labels only (6+ months)

observation_df, outcome_df = split_by_time_window(
    df, 
    observation_end=OBS_END,
    outcome_start=OUTCOME_START
)

check_data_leakage(observation_df, outcome_df)

log_info(f"Observation: {len(observation_df):,} transactions")
log_info(f"Outcome: {len(outcome_df):,} transactions")


# --- Step 3: Filter to Eligible Customers (Hybrid Criteria) ---
# Only include customers who have ESTABLISHED RELATIONSHIPS by observation cutoff
# This ensures we're modeling retention degradation, not acquisition failure
#
# Criteria 1 (Tenure): Customer's first purchase must be >= min_tenure_days 
#                      before observation ends (allows time for repeat behavior)
# Criteria 2 (Frequency): Customer must have >= min_frequency transactions
#                         (filters out one-off "tourist" buyers)
#
# Business Logic: Cannot predict churn for customers with no established pattern
# A customer who bought once 14 months ago and never returned is not "churning" -
# they never established a relationship in the first place

log_info("Applying cohort eligibility criteria...")

# Load eligibility criteria from config
MIN_TENURE_DAYS = config['features']['min_tenure_days']
MIN_FREQUENCY = config['features']['min_frequency']

obs_end_date = pd.to_datetime(OBS_END)
min_acquisition_date = obs_end_date - pd.Timedelta(days=MIN_TENURE_DAYS)

# Calculate customer acquisition dates and transaction counts
customer_stats = observation_df.groupby('customer_id').agg(
    first_purchase=('invoice_date', 'min'),
    n_transactions=('invoice', 'nunique')
)

# Apply hybrid filter
eligible_customers = customer_stats[
    (customer_stats['first_purchase'] <= min_acquisition_date) &  # Tenure >= 90 days
    (customer_stats['n_transactions'] >= MIN_FREQUENCY)            # Frequency >= 2
].index

# Log filtering breakdown for transparency
n_total = len(customer_stats)
n_tenure_fail = (customer_stats['first_purchase'] > min_acquisition_date).sum()
n_frequency_fail = (customer_stats['n_transactions'] < MIN_FREQUENCY).sum()
n_both_fail = (
    (customer_stats['first_purchase'] > min_acquisition_date) &
    (customer_stats['n_transactions'] < MIN_FREQUENCY)
).sum()

# Update observation dataframe to only eligible customers
observation_df = observation_df[
    observation_df['customer_id'].isin(eligible_customers)
]

n_eligible = len(eligible_customers)
filtered_out = n_total - n_eligible
filtered_pct = (filtered_out / n_total) * 100

log_info(f"Cohort Eligibility Results:")
log_info(f"  Total customers in observation: {n_total:,}")
log_info(f"  Eligible (meeting both criteria): {n_eligible:,}")
log_info(f"  Filtered out: {filtered_out:,} ({filtered_pct:.1f}%)")
log_info(f"    - Failed tenure (< {MIN_TENURE_DAYS} days): {n_tenure_fail:,}")
log_info(f"    - Failed frequency (< {MIN_FREQUENCY} txns): {n_frequency_fail:,}")
log_info(f"    - Failed both: {n_both_fail:,}")


# --- Step 4: Calculate RFM Metrics ---
# Recency: Days since last transaction (as of observation end)
# Frequency: Total number of unique invoices (purchases + returns)
# Monetary_Net: Net revenue (purchases minus returns, can be negative)
# Monetary_Gross: Purchases only (positive total_amount, always non-negative)

log_info("Calculating RFM metrics...")

obs_end_date = pd.to_datetime(OBS_END)

rfm = observation_df.groupby('customer_id').agg(
    recency=('invoice_date', lambda x: (obs_end_date - x.max()).days),
    frequency=('invoice', 'nunique'),  # Count unique invoices (purchases + returns)
    monetary_net=('total_amount', 'sum'),  # Net revenue (can be negative)
    monetary_gross=('total_amount', lambda x: x[x > 0].sum())  # Purchases only
).reset_index()


# --- Step 5: Calculate Behavioral Features ---
# Metrics that describe customer purchasing patterns and engagement levels

log_info("Calculating behavioral features...")

# --- Basket Size Metrics (Dual Metrics for B2B/B2C Signal) ---
# avg_items_per_basket: Total items per checkout event (engagement proxy)
# avg_units_per_line: Average quantity per product line (bulk buying intent)

# Calculate true basket size (sum quantities per invoice)
basket_sizes = observation_df[observation_df['quantity'] > 0].groupby(
    ['customer_id', 'invoice']
)['quantity'].sum()

basket_metrics = pd.DataFrame({
    'customer_id': basket_sizes.groupby('customer_id').groups.keys(),
    'avg_items_per_basket': basket_sizes.groupby('customer_id').mean(),
    'avg_units_per_line': observation_df[observation_df['quantity'] > 0].groupby(
        'customer_id'
    )['quantity'].mean()
}).reset_index(drop=True)

# --- Core Behavioral Features ---
behavioral = observation_df.groupby('customer_id').agg(
    unique_products=('stock_code', 'nunique'),
    first_purchase_date=('invoice_date', 'min'),
    last_purchase_date=('invoice_date', 'max')
).reset_index()

# Merge basket metrics
behavioral = behavioral.merge(basket_metrics, on='customer_id', how='left')

# --- Temporal Features ---
behavioral['days_as_customer'] = (
    (behavioral['last_purchase_date'] - behavioral['first_purchase_date']).dt.days
)

# Merge with RFM to get frequency for velocity calculation
behavioral = behavioral.merge(rfm[['customer_id', 'frequency']], on='customer_id')

# Purchase velocity with Laplace smoothing (+1 day to denominator)
# Prevents asymptotic explosion for same-day burst purchases
# Formula: frequency / ((days + 1) / 30)
behavioral['purchase_velocity'] = (
    behavioral['frequency'] / ((behavioral['days_as_customer'] + 1) / 30)
)

# Average days between purchases (with same smoothing)
behavioral['avg_days_between_purchases'] = (
    (behavioral['days_as_customer'] + 1) / behavioral['frequency']
)

# Drop temporary frequency column (will merge from RFM later)
behavioral = behavioral.drop(columns=['frequency'])


# --- Step 6: Calculate Return Metrics ---
# Track return behavior which may correlate with churn
# FIXED: Invoice-level return rate (dimensionally consistent with frequency)

log_info("Calculating return metrics...")

# Count return invoices (invoices with at least one return line item)
return_invoices = observation_df[observation_df['is_return']].groupby('customer_id')['invoice'].nunique()

# Total return amount (absolute value for interpretation)
return_amount = observation_df[observation_df['is_return']].groupby('customer_id')['total_amount'].sum().abs()

# Combine into returns dataframe
returns = pd.DataFrame({
    'customer_id': return_invoices.index,
    'n_return_invoices': return_invoices.values,
    'return_amount': return_amount.reindex(return_invoices.index, fill_value=0).values
}).reset_index(drop=True)

# Merge with RFM to get total frequency (invoices)
returns = returns.merge(rfm[['customer_id', 'frequency']], on='customer_id', how='right')

# Fill missing values (customers with no returns)
returns['n_return_invoices'] = returns['n_return_invoices'].fillna(0).astype(int)
returns['return_amount'] = returns['return_amount'].fillna(0)

# Calculate invoice-level return rate (bounded 0-1)
returns['return_rate'] = returns['n_return_invoices'] / returns['frequency']

# Boolean flag for any returns
returns['has_returns'] = returns['n_return_invoices'] > 0

# Drop temporary frequency column
returns = returns.drop(columns=['frequency'])


# --- Step 7: Define Churn Labels ---
# FIXED: Purchase-only definition (Option 1)
# Churned (1): No NEW PURCHASES in outcome window
# Retained (0): At least 1 purchase (quantity > 0) in outcome window
# Returns alone do not count as retention

log_info("Defining churn labels...")

# Get customers who made NEW PURCHASES in outcome window (quantity > 0)
purchase_customers = outcome_df[outcome_df['quantity'] > 0]['customer_id'].unique()

# Create churn labels for all eligible customers
churn_labels = pd.DataFrame({
    'customer_id': eligible_customers,
    'churned': ~pd.Series(eligible_customers).isin(purchase_customers).values
}).astype({'churned': int})

churn_rate = churn_labels['churned'].mean()
log_info(f"Churn rate: {churn_rate:.1%} ({churn_labels['churned'].sum():,} churned)")


# --- Step 8: Merge All Features ---
# Combine RFM, behavioral, return metrics, and churn labels

log_info("Merging feature sets...")

features = rfm.merge(behavioral, on='customer_id', how='left')
features = features.merge(returns, on='customer_id', how='left')
features = features.merge(churn_labels, on='customer_id', how='left')

# Fill missing basket metrics (customers with no valid purchase quantities)
features['avg_items_per_basket'] = features['avg_items_per_basket'].fillna(0)
features['avg_units_per_line'] = features['avg_units_per_line'].fillna(0)


# --- Step 9: Data Quality Checks ---
# Validate feature distributions and flag potential issues

log_info("Running data quality checks...")

# Check for negative recency (should not exist)
if (features['recency'] < 0).any():
    log_error("Negative recency values detected - check observation window logic")

# Check for zero frequency (all customers should have at least 1 transaction)
if (features['frequency'] == 0).any():
    log_warn(f"{(features['frequency'] == 0).sum()} customers with zero frequency")

# Check for missing churn labels
if features['churned'].isna().any():
    log_error(f"{features['churned'].isna().sum()} missing churn labels")

# Validate all customer_ids are unique
if features['customer_id'].duplicated().any():
    log_error("Duplicate customer_ids detected")

# Validate monetary metrics
if (features['monetary_gross'] < 0).any():
    log_error("Negative monetary_gross detected - should be impossible")

# Log customers with negative net monetary (serial returners)
neg_monetary = (features['monetary_net'] < 0).sum()
if neg_monetary > 0:
    log_info(f"{neg_monetary} customers with negative net monetary (serial returners)")


# --- Step 10: Generate Feature Report ---

report = generate_feature_report(features)
print(report)


# --- Step 11: Export Features ---

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

features.to_csv(OUTPUT_PATH, index=False)

log_info(f"Features exported: {OUTPUT_PATH}")
log_info(f"File size: {OUTPUT_PATH.stat().st_size / (1024**2):.2f} MB")
log_info("Feature engineering complete - ready for modeling")


# ===========================================================================
# FEATURE DICTIONARY (Embedded Documentation)
# ===========================================================================
# Features for ML Modeling:
# - customer_id: Unique identifier (not used in model, for tracking only)
# - recency: Days since last transaction (as of 2011-05-31)
# - frequency: Number of unique invoices (purchases + returns)
# - monetary_net: Net revenue (purchases minus returns, CAN BE NEGATIVE)
# - monetary_gross: Total purchase revenue (returns excluded, always ≥ 0)
# - avg_items_per_basket: Average total items per invoice (engagement proxy)
# - avg_units_per_line: Average quantity per product line (bulk buying signal)
# - unique_products: Number of distinct products purchased
# - first_purchase_date: Date of first transaction
# - last_purchase_date: Date of last transaction in observation window
# - days_as_customer: Span from first to last transaction
# - purchase_velocity: Transactions per 30 days (Laplace smoothed: freq/(days+1)*30)
# - avg_days_between_purchases: Average time between invoices (Laplace smoothed)
# - n_return_invoices: Count of invoices containing returns
# - return_amount: Total value of returned items (positive value)
# - return_rate: Proportion of invoices that are returns (0.0 to 1.0)
# - has_returns: Boolean flag (True if customer has any return invoices)
# - churned: Target variable (1=no new purchases in outcome, 0=purchased)
# ===========================================================================