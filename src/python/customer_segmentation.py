"""
E-Commerce Churn Prediction - Customer Segmentation
Purpose: Unsupervised clustering for business persona creation (RFM only)
Input: data/processed/customers_features.csv
Output: data/processed/customer_segments.csv
Architecture: K-Means on log-transformed RFM features.
               Log-transform (np.log1p) compresses monetary outliers before
               StandardScaler, preventing whale customers from collapsing
               Silhouette score to K=2.
               K search restricted to range(4, 8) — business utility bounds.
               Segment naming via rank-based composite RFM scorer —
               eliminates heuristic threshold collisions, guarantees
               unique names for every cluster regardless of K.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys
sys.path.append(str(Path(__file__).parent.parent))
from python.utils import log_info, log_warn


# --- Load Configuration ---

config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

FEATURES_PATH = Path(config['paths']['processed_data']) / "customers_features.csv"
OUTPUT_PATH = Path(config['paths']['processed_data']) / "customer_segments.csv"
RANDOM_STATE = config['modeling']['random_state']

log_info("Configuration loaded successfully")


# --- Step 1: Load Data and Select RFM Features ---

log_info("Loading customer features...")

df = pd.read_csv(FEATURES_PATH)

rfm_features = ['recency', 'frequency', 'monetary_gross']
X_rfm = df[rfm_features].copy()

log_info(f"Data loaded: {len(df):,} customers")
log_info(f"Clustering features: {rfm_features}")


# --- Step 2: Log-Transform RFM Features ---
# RFM data follows a right-skewed Pareto distribution (80/20 rule).
# Without transformation, extreme monetary outliers (e.g. £100k+ B2B whales)
# dominate Euclidean distance, collapsing Silhouette to K=2
# ("Whales" vs "Everyone Else") — mathematically optimal, strategically useless.
#
# np.log1p() = log(1 + x): safely handles zero values and compresses
# the long right tail into an approximately normal distribution,
# allowing K-Means to discover genuine behavioral clusters.

X_rfm_log = np.log1p(X_rfm)

log_info("RFM features log-transformed (np.log1p) to compress outlier skew")

for col in rfm_features:
    log_info(f"  {col}: raw_mean={X_rfm[col].mean():.1f}, "
             f"log_mean={X_rfm_log[col].mean():.3f}, "
             f"raw_max={X_rfm[col].max():.1f}, "
             f"log_max={X_rfm_log[col].max():.3f}")


# --- Step 3: Feature Scaling ---
# StandardScaler applied AFTER log-transform to center and scale
# the already-compressed distribution.

scaler = StandardScaler()
X_rfm_scaled = scaler.fit_transform(X_rfm_log)

log_info("Log-transformed RFM features standardized (StandardScaler)")


# --- Step 4: Determine Optimal K Within Business-Utility Bounds ---
# Search restricted to K=4 to K=7.
#
# Why not K=2 or K=3:
#   A marketing team cannot run targeted CRM strategies with fewer than
#   4 personas. Even post-log-transform, K=2/3 risk yielding "High Value"
#   vs "Low Value" — a binary split with no actionable granularity.
#
# Why not K=8+:
#   Beyond 7 clusters, personas become too granular for campaign execution
#   and difficult to explain to business stakeholders.
#
# The Silhouette score identifies the best mathematical configuration
# WITHIN the bounds of business utility. Algorithms serve the business.

log_info("Determining optimal K within business-utility range (K=4 to K=7)...")

inertias = []
silhouette_scores = []
K_range = range(4, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_rfm_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_rfm_scaled, kmeans.labels_))

log_info("K-Means evaluation (log-transformed + scaled):")
for k, inertia, sil_score in zip(K_range, inertias, silhouette_scores):
    log_info(f"  K={k}: Inertia={inertia:.2f}, Silhouette={sil_score:.4f}")

optimal_k = list(K_range)[np.argmax(silhouette_scores)]
log_info(f"Optimal K selected: {optimal_k} (highest silhouette within K=4–7)")


# --- Step 5: Train Final K-Means Model ---

log_info(f"Training K-Means with K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
df['segment'] = kmeans_final.fit_predict(X_rfm_scaled)

log_info(f"Clustering complete: {optimal_k} segments created")


# --- Step 6: Profile Segments ---

log_info("Profiling customer segments...")

segment_profiles = df.groupby('segment').agg({
    'recency': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'monetary_gross': ['mean', 'median'],
    'churned': 'mean',
    'customer_id': 'count'
}).round(2)

segment_profiles.columns = [
    'recency_mean', 'recency_median',
    'frequency_mean', 'frequency_median',
    'monetary_mean', 'monetary_median',
    'churn_rate', 'customer_count'
]

segment_profiles = segment_profiles.reset_index()


# --- Step 7: Name Segments via Rank-Based Composite RFM Scorer ---
# ARCHITECTURE NOTE:
# The previous heuristic approach used global quantile thresholds + if/elif
# chains to assign names. This caused naming collisions when K-Means clusters
# didn't perfectly satisfy the specific threshold combinations, causing
# multiple distinct clusters to fall into the final `else` bucket.
#
# Fix: Compute a composite RFM score per segment by min-max normalizing
# each metric across segments, then summing. Recency is inverted (lower
# days = more recent = better). Segments are ranked by composite score and
# assigned names from an ordered pool — guaranteeing uniqueness for every
# cluster regardless of K.

def assign_segment_names(profiles):
    """
    Rank segments by composite RFM score and assign unique business names.

    Composite score = recency_score (inverted) + frequency_score + monetary_score
    Each component is min-max normalized to [0, 1] across all segments.
    Rank 1 (highest composite) → 'Champions'
    Rank N (lowest composite)  → 'Lost Customers'
    """
    p = profiles.copy()

    def minmax(series):
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / rng

    # Lower recency = purchased more recently = better, so invert
    recency_score  = 1 - minmax(p['recency_mean'])
    frequency_score = minmax(p['frequency_mean'])
    monetary_score  = minmax(p['monetary_mean'])

    p['composite_score'] = recency_score.values + frequency_score.values + monetary_score.values
    p = p.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # Ordered name pool: best persona → worst persona
    # Pool is intentionally larger than max K=7 for safety
    name_pool = [
        "Champions",          # Rank 1: recent, frequent, high value
        "Loyal Customers",    # Rank 2: frequent and valuable, slightly less recent
        "Potential Loyalists",# Rank 3: recent but lower frequency/spend
        "At Risk",            # Rank 4: used to be good, now lapsing
        "Need Attention",     # Rank 5: mid-tier, declining engagement
        "Hibernating",        # Rank 6: long inactive, low value
        "Lost Customers"      # Rank 7: not recent, low frequency, low value
    ]

    p['segment_name'] = [name_pool[i] for i in range(len(p))]
    return p

segment_profiles = assign_segment_names(segment_profiles)

log_info("Segment names assigned via rank-based composite RFM scorer")
for _, row in segment_profiles.iterrows():
    log_info(f"  Rank composite={row['composite_score']:.3f} → {row['segment_name']}")

# Map K-Means cluster IDs back to segment names in main dataframe
segment_name_map = segment_profiles.set_index('segment')['segment_name'].to_dict()
df['segment_name'] = df['segment'].map(segment_name_map)


# --- Step 8: Generate Business Recommendations per Segment ---

recommendations = {
    "Champions":           "Reward loyalty. Offer VIP programs, early access to new products, and exclusive discounts.",
    "Loyal Customers":     "Upsell and cross-sell. Recommend premium products and bundles.",
    "Potential Loyalists": "Build the habit. Targeted follow-up campaigns and loyalty program invitations.",
    "At Risk":             "Re-engagement campaigns. Send win-back offers, surveys, and personalized recommendations.",
    "Need Attention":      "Reactivate with value. Time-limited offers and product recommendations based on past purchases.",
    "Hibernating":         "Reactivation campaigns. Limited-time offers and product updates.",
    "Lost Customers":      "Aggressive win-back. Deep discounts, apology campaigns, or let them go if unprofitable."
}

segment_profiles['recommendation'] = segment_profiles['segment_name'].map(recommendations)


# --- Step 9: Display Segment Summary ---

log_info("\n" + "="*77)
log_info("CUSTOMER SEGMENT PROFILES")
log_info("="*77)

for _, row in segment_profiles.iterrows():
    log_info(f"\nSegment: {row['segment_name']}")
    log_info(f"  Size: {row['customer_count']:,} customers")
    log_info(f"  Churn Rate: {row['churn_rate']:.1%}")
    log_info(f"  Avg Recency: {row['recency_mean']:.0f} days")
    log_info(f"  Avg Frequency: {row['frequency_mean']:.1f} transactions")
    log_info(f"  Avg Monetary: £{row['monetary_mean']:,.2f}")
    log_info(f"  Recommendation: {row['recommendation']}")

log_info("="*77 + "\n")


# --- Step 10: Save Segmented Data ---

df_output = df[['customer_id', 'segment', 'segment_name', 'recency', 'frequency',
                'monetary_gross', 'churned']].copy()
df_output.to_csv(OUTPUT_PATH, index=False)
log_info(f"Segmented customers saved: {OUTPUT_PATH}")

profile_path = OUTPUT_PATH.parent / "segment_profiles.csv"
segment_profiles.to_csv(profile_path, index=False)
log_info(f"Segment profiles saved: {profile_path}")

log_info("Customer segmentation complete - ready for dashboard")