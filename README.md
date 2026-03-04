# 🛒 E-Commerce Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A production-grade, end-to-end churn prediction system that identifies at-risk customers and segments a customer base into actionable marketing personas for an e-commerce retailer.**
> This system models real-world customer lifetime value and disengagement behaviour, handling a Pareto-distributed transactional dataset of 1,067,371 rows using a dual-pipeline architecture in R and Python — culminating in a live, interactive Streamlit dashboard.

<div align="center">

🌐 **[Live Dashboard → ecommerce-churn-prediction-teb2043.streamlit.app](https://ecommerce-churn-prediction-teb2043.streamlit.app/)**

</div>

---

## 🛠️ Tech Stack

- **Core:** Python 3.10+, R 4.x, YAML
- **ML & Data:** Scikit-Learn 1.8, XGBoost, pandas, numpy, scipy
- **Visualisation & Dashboard:** Streamlit, Plotly
- **Data Cleaning:** R (dplyr, lubridate, janitor, readxl)
- **Concepts:** Supervised Binary Classification, Unsupervised Clustering (K-Means), RFM Analysis, Feature Engineering, Cross-Validation Pipeline Architecture, Hyperparameter Optimisation

---

## ⚙️ Core Architecture & Features

**Dual-Language Data Pipeline:** Built a two-stage cleaning and engineering pipeline — R handles raw Excel ingestion, deduplication with weighted average pricing, and administrative code filtering; Python handles time-window splitting, cohort filtering, and all 14 feature calculations. The strict stage separation means each layer can be swapped or retrained independently.

**Leakage-Free Model Training:** Wrapped `StandardScaler` inside a scikit-learn `Pipeline` alongside `LogisticRegression`, ensuring the scaler is fitted exclusively on each CV fold's training split during `RandomizedSearchCV`. This eliminates the methodological flaw where pre-scaling contaminates validation fold statistics — a subtle but critical correctness fix. Three classifiers were trained (Logistic Regression, Random Forest, XGBoost), all achieving Test ROC-AUC between 0.776 and 0.780.

**Log-Transformed RFM Segmentation:** Applied `np.log1p()` before `StandardScaler` on RFM features prior to K-Means clustering. Without this step, Pareto-distributed monetary data collapses the Silhouette-optimal K to 2 — a statistically valid but strategically useless "Whales vs Everyone Else" split. The transform recovers four genuinely distinct behavioural personas (Champions 5% churn → At Risk 59% churn), each paired with a concrete CRM recommendation.

**Production-Serialised Artefacts:** All trained models are saved as `.pkl` files. The Logistic Regression `.pkl` is the full Pipeline object (scaler embedded), meaning deployment requires zero preprocessing code — call `.predict_proba()` directly on raw unscaled input. A `feature_names.json` and `evaluation_metrics.json` (containing confusion matrices, all test scores, and CV scores) are saved alongside for dashboard consumption without any model retraining.

**Three-Page Interactive Dashboard:** A Streamlit app with `@st.cache_resource` model loading serving three audiences — an Executive Overview page with KPI cards, a 3D RFM scatter plot, persona radar charts, and a one-click At Risk customer CSV export; a Model Diagnostics page with confusion matrix heatmaps and feature importance; and a Prediction Engine page with a live churn risk gauge and all three models scoring any selected customer ID in real time.

---

## 🎓 Engineering Takeaways

**Preprocessing Order is Business Logic:** The R cleaning pipeline enforces strict step sequencing — price filtering *must* precede duplicate aggregation. Reversing this order allows negative prices to corrupt the weighted average calculation silently. Similarly, dplyr's top-to-bottom `summarise()` evaluation means price must be computed before quantity to avoid destroying the source vector. These are not style choices; they are correctness constraints.

**Algorithm Constraints Should Serve the Business:** The K-Means K search range was deliberately restricted to K=4–7 rather than the unconstrained K=2–10. This is a business-utility constraint, not a mathematical one — a CRM team cannot execute targeted campaigns against fewer than four personas. Letting the Silhouette score freely select K=2 would have been mathematically defensible and practically useless. Algorithms serve the business, not the other way around.

**Sampling Distributions Matter for Hyperparameter Search:** Regularisation strength `C` and `learning_rate` operate on logarithmic magnitudes. Using `scipy.stats.uniform(0.001, 100)` allocates 99% of `RandomizedSearchCV` iterations to large values where model behaviour is indistinguishable. Switching to `loguniform` distributes the search budget proportionally across all orders of magnitude — the same 15 iterations, meaningfully more useful results.

**Deprecation Warnings Are Technical Debt:** The sklearn `penalty=` argument was deprecated in v1.8. The fix is not suppressing warnings — it is removing `penalty='elasticnet'` entirely from the `LogisticRegression` constructor, since passing `l1_ratio` via the hyperparameter grid is sufficient for the modern API to infer regularisation behaviour. Shipping code with known deprecation warnings in a production pipeline is a liability, not a minor inconvenience.

---

## 🚀 Future Roadmap

- [ ] Save `y_pred_proba` arrays to `evaluation_metrics.json` to enable full ROC curve visualisation in the dashboard without model retraining
- [ ] Implement a periodic retraining scheduler that re-executes the full pipeline on fresh transaction data, keeping churn signals current
- [ ] Add a propensity-to-respond score alongside churn probability to prioritise reactivation spend on customers most likely to re-engage
- [ ] Explore survival analysis (Cox Proportional Hazards) to model time-to-churn rather than a binary label, providing a risk timeline for CRM scheduling
- [ ] Containerise the full application using Docker for one-command local deployment across environments
- [ ] Build separate churn models per customer channel (B2B wholesale vs B2C retail) using the `avg_units_per_line` signal as a soft channel classifier

---

## 🖥️ Local Execution

**Prerequisites:** Python 3.10+, R 4.x, RStudio (for the cleaning script), the UCI Online Retail II dataset placed at `data/raw/online_retail_ii.xlsx`.

### 1. Clone the repository

```bash
git clone https://github.com/MaheshV-13/ecommerce-churn-prediction
cd ecommerce-churn-prediction
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install R dependencies

Open RStudio and run:

```r
install.packages(c("tidyverse", "readxl", "janitor", "lubridate", "yaml"))
```

### 4. Run the data pipeline (in order)

```bash
# Step 1 — R data cleaning (run in RStudio or Rscript)
Rscript src/R/data_cleaning.R

# Step 2 — Python feature engineering
python src/python/feature_engineering.py

# Step 3 — Model training
python src/python/model_training.py

# Step 4 — Customer segmentation
python src/python/customer_segmentation.py
```

### 5. Launch the dashboard

```bash
streamlit run src/python/dashboard.py
```

---

## 📁 Project Structure

```
ecommerce-churn-prediction/
├── config/
│   └── config.yaml                  # All paths, date windows, and model parameters
├── data/
│   ├── raw/
│   │   └── online_retail_ii.xlsx    # UCI source dataset (not tracked in git)
│   ├── interim/
│   │   └── cleaned_retail_data.csv  # After R cleaning: 783,684 rows
│   └── processed/
│       ├── customers_features.csv   # After feature engineering: 3,463 customers
│       ├── customer_segments.csv    # K-Means output with segment labels
│       └── segment_profiles.csv     # Aggregate RFM stats per segment
├── models/
│   ├── logistic_regression.pkl      # Full Pipeline (StandardScaler + LR embedded)
│   ├── random_forest.pkl            # Champion model — Test ROC-AUC 0.7798
│   ├── xgboost.pkl
│   ├── evaluation_metrics.json      # All test scores, CV scores, confusion matrices
│   └── feature_names.json           # Ordered list of 14 training features
├── src/
│   ├── R/
│   │   ├── data_cleaning.R          # Production R cleaning script
│   │   └── utils.R
│   └── python/
│       ├── feature_engineering.py   # 14-feature engineering pipeline
│       ├── model_training.py        # Pipeline + loguniform + RandomizedSearchCV
│       ├── customer_segmentation.py # Log-transform K-Means, rank-based naming
│       ├── dashboard.py             # Three-page Streamlit dashboard
│       └── utils.py
└── reports/
    └── G2 proposal.pdf                 
```

---

## 📊 Results Summary

| Model | CV ROC-AUC | Test ROC-AUC | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.7895 | 0.7767 | 0.6372 |
| **Random Forest ★** | **0.7877** | **0.7798** | **0.6478** |
| XGBoost | 0.7915 | 0.7792 | 0.6390 |

| Segment | Size | Churn Rate | Avg Monetary |
|---|---|---|---|
| Champions | 499 | 5% | £13,986 |
| Loyal Customers | 584 | 23% | £1,668 |
| Potential Loyalists | 970 | 32% | £2,667 |
| At Risk | 1,410 | 59% | £615 |

---
