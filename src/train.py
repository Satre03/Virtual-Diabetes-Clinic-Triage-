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
RANDOM_STATE = 42

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

# --- Save artifacts ---
joblib.dump(model, ART_DIR / "model.joblib")
(ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

print(f"✅ Training complete. RMSE={rmse:.4f}")
print("Artifacts saved in", ART_DIR)
