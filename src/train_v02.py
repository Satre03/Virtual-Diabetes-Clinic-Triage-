from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
RISK_QUANTILE = 0.75  # top 25% flagged high-risk
MODEL_VERSION = "0.2.0"

print(f"--- Training model version {MODEL_VERSION} ---")

# Load data
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]
features = list(X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Define model pipeline
print("Using model: RandomForestRegressor (v0.2)")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"Test RMSE for v{MODEL_VERSION}: {rmse:.4f}")

# High-risk flag metrics
threshold = float(np.quantile(y_train, RISK_QUANTILE))
y_true_flag = (y_test.values >= threshold).astype(int)
y_pred_flag = (y_pred >= threshold).astype(int)
precision = float(precision_score(y_true_flag, y_pred_flag, zero_division=0))
recall = float(recall_score(y_true_flag, y_pred_flag, zero_division=0))
print(
    "High-risk flag @ train "
    f"{int(RISK_QUANTILE * 100)}th pct (thr={threshold:.3f}) → "
    f"precision={precision:.3f}, recall={recall:.3f}"
)

# Save artifacts
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, artifacts_dir / "model.joblib")

# Save metrics
metrics = {
    "version": MODEL_VERSION,
    "rmse": rmse,
    "risk_threshold": threshold,
    "precision_flag": precision,
    "recall_flag": recall
}
(artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

# Save metadata
meta = {
    "pipeline": "scaler+random_forest",
    "version": MODEL_VERSION,
    "features": features,
    "random_seed": RANDOM_SEED,
    "risk_quantile": RISK_QUANTILE,
    "trained_at": datetime.now(timezone.utc).isoformat()
}
(artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# Print summary
print(json.dumps({
    "rmse": rmse,
    "precision": precision,
    "recall": recall
}, indent=2))
