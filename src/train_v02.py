import json
import joblib
import os
from datetime import date
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
RISK_QUANTILE = 0.75  # top 25% flagged high-risk

print("--- Training model version 0.2 ---")

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]
features = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

print("Using model: RandomForestRegressor (v0.2)")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        max_features="sqrt",
        random_state=RANDOM_SEED
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"Test RMSE for v0.2: {rmse:.4f}")

threshold = float(np.quantile(y_train, RISK_QUANTILE))
y_true_flag = (y_test.values >= threshold).astype(int)
y_pred_flag = (y_pred >= threshold).astype(int)
precision = float(precision_score(y_true_flag, y_pred_flag, zero_division=0))
recall = float(recall_score(y_true_flag, y_pred_flag, zero_division=0))
print(
    "High-risk flag @ train "
    f"{int(RISK_QUANTILE*100)}th pct (thr={threshold:.3f}) → "
    f"precision={precision:.3f}, recall={recall:.3f}"
)

os.makedirs("models", exist_ok=True)

model_path = "models/model_v0.2.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

feature_path = "models/feature_list.json"
with open(feature_path, "w") as f:
    json.dump(features, f)
print(f"Features saved to {feature_path}")

calibration_path = "models/calibration_v0.2.json"
with open(calibration_path, "w") as f:
    json.dump({"risk_threshold": threshold, "quantile": RISK_QUANTILE}, f, indent=4)
print(f"Calibration saved to {calibration_path}")

metrics_v02 = {
    "version": "0.2",
    "rmse": rmse,
    "risk_threshold": threshold,
    "precision_flag": precision,
    "recall_flag": recall
}
with open("metrics_v02.json", "w") as f:
    json.dump(metrics_v02, f, indent=4)
print("Metrics saved to metrics_v02.json")

if os.path.exists("
