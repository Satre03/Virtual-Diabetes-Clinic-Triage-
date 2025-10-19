import os
import json
import joblib
import numpy as np  # noqa: F401  # kept for later numeric operations
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Setup ---
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = Path("models")  # new folder for Docker
MODELS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

# --- Load dataset ---
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# --- Preprocess + train baseline model ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
preds = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, preds, squared=False)

metrics = {"rmse": rmse}
meta = {"model": "LinearRegression", "random_state": RANDOM_STATE}

# --- Save artifacts (for CI reproducibility) ---
joblib.dump(model, ART_DIR / "model.joblib")
(ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

# --- Save model and features for Docker image ---
joblib.dump(model, MODELS_DIR / f"model_v{MODEL_VERSION}.joblib")
(MODELS_DIR / "feature_list.json").write_text(json.dumps(list(X.columns), indent=2))

print(f"Training complete. RMSE={rmse:.4f}")
print("Artifacts saved in", ART_DIR)
print("Model + features saved in", MODELS_DIR)
