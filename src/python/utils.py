"""
Utility Functions for Feature Engineering
Purpose: Reusable functions for logging, validation, and data quality checks
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple


# --- Logging Functions ---

def log_info(message: str) -> None:
    """Print INFO level log message."""
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    """Print WARN level log message."""
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    """Print ERROR level log message."""
    print(f"[ERROR] {message}")


# --- Data Validation Functions ---

def validate_date_column(df: pd.DataFrame, col_name: str) -> None:
    """
    Validate that a date column is properly formatted and contains no NaT values.
    Raises ValueError if validation fails.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe")
    
    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        raise ValueError(f"Column '{col_name}' is not datetime type")
    
    nat_count = df[col_name].isna().sum()
    if nat_count > 0:
        log_warn(f"{nat_count} NaT values found in {col_name}")


def validate_no_missing_values(df: pd.DataFrame, 
                                critical_columns: list) -> None:
    """
    Check for missing values in critical columns and log warnings.
    """
    missing = df[critical_columns].isna().sum()
    missing = missing[missing > 0]
    
    if len(missing) > 0:
        log_warn(f"Missing values in critical columns: {missing.to_dict()}")


def check_data_leakage(observation_df: pd.DataFrame, 
                       outcome_df: pd.DataFrame) -> None:
    """
    Verify no temporal overlap between observation and outcome windows.
    Raises ValueError if leakage detected.
    """
    obs_max = observation_df['invoice_date'].max()
    outcome_min = outcome_df['invoice_date'].min()
    
    if obs_max >= outcome_min:
        raise ValueError(
            f"Data leakage detected: observation max ({obs_max}) >= "
            f"outcome min ({outcome_min})"
        )
    
    log_info("No temporal data leakage detected")


# --- Feature Quality Report ---

def generate_feature_report(df: pd.DataFrame) -> str:
    """
    Generate summary statistics for engineered features.
    Returns formatted report string.
    """
    report = [
        "\n" + "=" * 77,
        "FEATURE ENGINEERING REPORT",
        "=" * 77,
        f"\nDataset Dimensions: {df.shape[0]:,} customers × {df.shape[1]} features",
        ""
    ]
    
    # RFM summary (updated column names)
    if 'recency' in df.columns:
        report.extend([
            "RFM Metrics:",
            f"  Recency (days): min={df['recency'].min():.0f}, "
            f"max={df['recency'].max():.0f}, mean={df['recency'].mean():.1f}",
            f"  Frequency: min={df['frequency'].min():.0f}, "
            f"max={df['frequency'].max():.0f}, mean={df['frequency'].mean():.1f}",
            f"  Monetary Net (£): min={df['monetary_net'].min():.2f}, "
            f"max={df['monetary_net'].max():.2f}, mean={df['monetary_net'].mean():.2f}",
            f"  Monetary Gross (£): min={df['monetary_gross'].min():.2f}, "
            f"max={df['monetary_gross'].max():.2f}, mean={df['monetary_gross'].mean():.2f}",
            ""
        ])
    
    # Churn distribution
    if 'churned' in df.columns:
        churn_dist = df['churned'].value_counts()
        churn_rate = (churn_dist.get(1, 0) / len(df)) * 100
        report.extend([
            "Churn Distribution:",
            f"  Churned (1): {churn_dist.get(1, 0):,} ({churn_rate:.1f}%)",
            f"  Retained (0): {churn_dist.get(0, 0):,} ({100-churn_rate:.1f}%)",
            ""
        ])
    
    # Return statistics
    if 'has_returns' in df.columns:
        return_pct = (df['has_returns'].sum() / len(df)) * 100
        avg_return_rate = df['return_rate'].mean()
        report.extend([
            "Return Behavior:",
            f"  Customers with returns: {df['has_returns'].sum():,} ({return_pct:.1f}%)",
            f"  Average return rate: {avg_return_rate:.1%}",
            ""
        ])
    
    # Serial returner statistics (negative net monetary)
    if 'monetary_net' in df.columns:
        neg_monetary = (df['monetary_net'] < 0).sum()
        if neg_monetary > 0:
            neg_pct = (neg_monetary / len(df)) * 100
            report.extend([
                "Serial Returners:",
                f"  Customers with negative net value: {neg_monetary:,} ({neg_pct:.1f}%)",
                ""
            ])
    
    report.append("=" * 77 + "\n")
    
    return "\n".join(report)


# --- Time Window Utilities ---

def split_by_time_window(df: pd.DataFrame, 
                         observation_end: str,
                         outcome_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split transactions into observation and outcome windows.
    
    Parameters:
    -----------
    df : DataFrame with 'invoice_date' column
    observation_end : End date of observation window (exclusive), format 'YYYY-MM-DD'
    outcome_start : Start date of outcome window (inclusive), format 'YYYY-MM-DD'
    
    Returns:
    --------
    observation_df, outcome_df : Tuple of DataFrames
    """
    obs_end = pd.to_datetime(observation_end)
    out_start = pd.to_datetime(outcome_start)
    
    observation_df = df[df['invoice_date'] < obs_end].copy()
    outcome_df = df[df['invoice_date'] >= out_start].copy()
    
    log_info(f"Observation window: {observation_df['invoice_date'].min()} to "
             f"{observation_df['invoice_date'].max()}")
    log_info(f"Outcome window: {outcome_df['invoice_date'].min()} to "
             f"{outcome_df['invoice_date'].max()}")
    
    return observation_df, outcome_df