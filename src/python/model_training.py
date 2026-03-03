"""
E-Commerce Churn Prediction - Model Training Pipeline
Purpose: Train supervised ML models for churn prediction using all features
Input: data/processed/customers_features.csv
Output: models/*.pkl, models/evaluation_metrics.json
Architecture:
  - Logistic Regression wrapped in sklearn Pipeline (scaler + model)
    → eliminates CV data leakage from pre-scaling
    → loguniform distribution for C (log-scale regularization search)
  - Random Forest: RandomizedSearchCV, class_weight='balanced'
  - XGBoost: RandomizedSearchCV, scale_pos_weight, loguniform learning_rate
  - No SMOTE: 37.7% churn rate is not severe imbalance
"""

import pandas as pd
import numpy as np
import yaml
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from scipy.stats import uniform, randint, loguniform

import sys
sys.path.append(str(Path(__file__).parent.parent))
from python.utils import log_info, log_warn, log_error


# --- Load Configuration ---

config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

FEATURES_PATH = Path(config['paths']['processed_data']) / "customers_features.csv"
MODELS_PATH = Path(config['paths']['models'])
MODELS_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = config['modeling']['random_state']
TEST_SIZE    = config['modeling']['test_size']
CV_FOLDS     = config['modeling']['cv_folds']

log_info("Configuration loaded successfully")


# --- Step 1: Load and Prepare Data ---

log_info("Loading customer features...")

df = pd.read_csv(FEATURES_PATH, parse_dates=['first_purchase_date', 'last_purchase_date'])

log_info(f"Data loaded: {len(df):,} customers, {df.shape[1]} columns")

drop_cols = ['customer_id', 'first_purchase_date', 'last_purchase_date', 'churned']
X = df.drop(columns=drop_cols)
y = df['churned']

feature_names = X.columns.tolist()

log_info(f"Features for modeling: {len(feature_names)} predictors")
log_info(f"Target distribution: {y.value_counts().to_dict()}")


# --- Step 2: Train-Test Split ---
# Stratified split to preserve class distribution in both sets.
# NOTE: No global scaling step here. Logistic Regression scaling is handled
# strictly inside the Pipeline (Step 4), preventing CV data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

log_info(f"Train set: {len(X_train):,} samples")
log_info(f"Test set:  {len(X_test):,} samples")


# --- Step 3: Define Hyperparameter Distributions ---
#
# KEY FIXES vs. previous version:
#
# Fix 1 — loguniform for C (Logistic Regression):
#   Regularization strength operates on logarithmic magnitudes.
#   uniform(0.001, 100) wastes 99% of iterations on large values where
#   the model behaves identically. loguniform(0.001, 100) allocates
#   compute proportionally across all orders of magnitude (0.001 → 100),
#   yielding a more accurate search in the same 15 iterations.
#
# Fix 2 — l1_ratio replaces penalty (Logistic Regression):
#   penalty='l1'/'l2' was deprecated in sklearn v1.8, removed in v1.10.
#   The modern API uses solver='saga' with penalty='elasticnet' and tunes
#   l1_ratio: 0.0 = pure L2, 1.0 = pure L1, (0,1) = ElasticNet blend.
#   This silences all FutureWarnings and future-proofs the pipeline.
#
# Fix 3 — loguniform for learning_rate (XGBoost):
#   Same logarithmic reasoning as C. Learning rates between 0.01 and 0.1
#   are the most sensitive region. uniform(0.01, 0.3) undersamples this
#   critical zone. loguniform(0.01, 0.3) allocates search budget correctly.

log_info("Defining hyperparameter distributions...")

# Logistic Regression (Pipeline params prefixed with 'model__')
param_dist_lr = {
    'model__C':        loguniform(0.001, 100),   # Log-scale: equal budget per order of magnitude
    'model__l1_ratio': uniform(0, 1),             # ElasticNet blend: 0=L2, 1=L1
}

# Random Forest
param_dist_rf = {
    'n_estimators':      randint(50, 300),
    'max_depth':         randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf':  randint(1, 10),
    'max_features':      ['sqrt', 'log2'],
    'class_weight':      ['balanced']
}

# XGBoost
param_dist_xgb = {
    'n_estimators':     randint(50, 300),
    'max_depth':        randint(3, 15),
    'learning_rate':    loguniform(0.01, 0.3),    # Log-scale: critical region 0.01–0.1
    'subsample':        uniform(0.6, 0.4),         # [0.6, 1.0]
    'colsample_bytree': uniform(0.6, 0.4),         # [0.6, 1.0]
    'gamma':            uniform(0, 5),
    'reg_alpha':        uniform(0, 1),
    'reg_lambda':       uniform(1, 10)
}


# --- Step 4: Train Models with RandomizedSearchCV ---

log_info("Training models with RandomizedSearchCV (15 iterations each)...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models     = {}
best_params = {}
cv_scores  = {}

# --- Model 1: Logistic Regression (Pipeline) ---
# ARCHITECTURE NOTE: 
# The scaler is embedded inside the Pipeline alongside the model.
# RandomizedSearchCV splits X_train (raw, unscaled) into CV folds, then
# fits the pipeline on each training fold — meaning the scaler's fit()
# only ever sees training fold data. The validation fold is transformed
# using those training-fold statistics only.
# This is the correct methodology. The previous approach of calling
# scaler.fit_transform(X_train) globally before CV contaminated every
# validation fold with the full training set's mean and variance.

log_info("Training Logistic Regression (Pipeline: StandardScaler → LogisticRegression)...")
start_time = datetime.now()

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        solver='saga',
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

lr_search = RandomizedSearchCV(
    lr_pipeline,
    param_dist_lr,
    n_iter=15,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)
lr_search.fit(X_train, y_train)    # Raw unscaled data — Pipeline handles scaling internally

models['logistic_regression']     = lr_search.best_estimator_   # Full pipeline saved
best_params['logistic_regression'] = lr_search.best_params_
cv_scores['logistic_regression']  = lr_search.best_score_

elapsed = (datetime.now() - start_time).total_seconds()
log_info(f"  Best CV ROC-AUC: {lr_search.best_score_:.4f} ({elapsed:.1f}s)")
log_info(f"  Best params: C={lr_search.best_params_['model__C']:.4f}, "
         f"l1_ratio={lr_search.best_params_['model__l1_ratio']:.4f}")


# --- Model 2: Random Forest ---

log_info("Training Random Forest...")
start_time = datetime.now()

rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf_search = RandomizedSearchCV(
    rf, param_dist_rf,
    n_iter=15,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)
rf_search.fit(X_train, y_train)

models['random_forest']     = rf_search.best_estimator_
best_params['random_forest'] = rf_search.best_params_
cv_scores['random_forest']  = rf_search.best_score_

elapsed = (datetime.now() - start_time).total_seconds()
log_info(f"  Best CV ROC-AUC: {rf_search.best_score_:.4f} ({elapsed:.1f}s)")


# --- Model 3: XGBoost ---

log_info("Training XGBoost...")
start_time = datetime.now()

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)
xgb_search = RandomizedSearchCV(
    xgb, param_dist_xgb,
    n_iter=15,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)
xgb_search.fit(X_train, y_train)

models['xgboost']     = xgb_search.best_estimator_
best_params['xgboost'] = xgb_search.best_params_
cv_scores['xgboost']  = xgb_search.best_score_

elapsed = (datetime.now() - start_time).total_seconds()
log_info(f"  Best CV ROC-AUC: {xgb_search.best_score_:.4f} ({elapsed:.1f}s)")


# --- Step 5: Evaluate on Test Set ---
# All models receive raw X_test.
# The LR pipeline applies its internally fitted scaler automatically.
# RF and XGBoost operate directly on unscaled data as before.

log_info("Evaluating models on test set...")

evaluation_results = {}

for model_name, model in models.items():
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    evaluation_results[model_name] = {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall':    float(recall_score(y_test, y_pred)),
        'f1_score':  float(f1_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_pred_proba)),
        'cv_roc_auc': float(cv_scores[model_name]),
        'best_params': {
            k: (int(v) if isinstance(v, (np.integer, np.int64)) else
                float(v) if isinstance(v, (np.floating, np.float64)) else v)
            for k, v in best_params[model_name].items()
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    log_info(f"{model_name.replace('_', ' ').title()}:")
    log_info(f"  Test ROC-AUC: {evaluation_results[model_name]['roc_auc']:.4f}")
    log_info(f"  Test F1-Score: {evaluation_results[model_name]['f1_score']:.4f}")


# --- Step 6: Save Models and Metrics ---
# NOTE: The saved logistic_regression.pkl is the full Pipeline object
# (StandardScaler + LogisticRegression). A separate scaler.pkl is no longer
# needed — the pipeline's predict() and predict_proba() methods apply
# scaling automatically, making deployment simpler and safer.

log_info("Saving trained models...")

for model_name, model in models.items():
    model_path = MODELS_PATH / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    log_info(f"  {model_name}: {model_path}")

metrics_path = MODELS_PATH / "evaluation_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(evaluation_results, f, indent=2)
log_info(f"Evaluation metrics saved: {metrics_path}")

feature_info = {'feature_names': feature_names, 'n_features': len(feature_names)}
features_path = MODELS_PATH / "feature_names.json"
with open(features_path, 'w') as f:
    json.dump(feature_info, f, indent=2)


# --- Step 7: Model Comparison Summary ---

log_info("\n" + "="*77)
log_info("MODEL COMPARISON SUMMARY")
log_info("="*77)

comparison_df = pd.DataFrame(evaluation_results).T
comparison_df = comparison_df[['cv_roc_auc', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']]
comparison_df.columns = ['CV ROC-AUC', 'Test ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

print(comparison_df.to_string())

best_model = comparison_df['Test ROC-AUC'].idxmax()
log_info(f"\nBest Model (by Test ROC-AUC): {best_model.replace('_', ' ').title()}")
log_info("="*77 + "\n")

log_info("Model training complete - ready for segmentation and dashboard")