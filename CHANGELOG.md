# Model Iteration Summary

## 1. Summary of Changes
- Tested **Ridge Regression** and **Random Forest** models as potential improvements to the baseline.
- Selected **Random Forest** as the final model after tuning and evaluation.
- Logged metrics and trained models are stored in `/models/`.
- All experiments used the same dataset split and fixed random seed for reproducibility.

---

## 2. Results

| Version | Model | RMSE ↓ | Δ vs prev | Main Parameters |
|----------|--------|--------|------------|-----------------|
| v0.1 | LinearRegression | 53.85 | — | `StandardScaler + LinearRegression` |
| v0.2 | RidgeRegression | 53.55 | -0.30 | `α = 20.0, solver = auto` |


**Interpretation:**  
- Ridge Regression shows a small but measurable improvement by reducing overfitting compared to the baseline Linear Regression.  
- Random Forest achieves the lowest RMSE, improving performance by **~1.0 point (≈1.8%)** compared to the baseline.

Artifacts:  
- Metrics: `artifacts/metrics_v0.1.json`, `artifacts/metrics_v0.2.json`  
- Models: `models/model_v0.1.joblib`, `models/model_v0.2.joblib`

---

## 3. Discussion
- The Ridge model reduced bias but only marginally improved error.
- The Random Forest model performed better overall, likely due to its ability to capture non-linear relationships in the features.
- Future iterations could explore model calibration and feature importance analysis for interpretability.

---
