import json
import joblib
import os
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

print("--- Training model version 0.1 ---")

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]
features = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

print("Using model: LinearRegression (v0.1)")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"Test RMSE for v0.1: {rmse:.4f}")

os.makedirs("models", exist_ok=True)

model_path = "models/model_v0.1.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

feature_path = "models/feature_list.json"
with open(feature_path, "w") as f:
    json.dump(features, f)
print(f"Features saved to {feature_path}")

metrics_v01 = {"version": "0.1", "rmse": rmse}
with open("metrics_v01.json", "w") as f:
    json.dump(metrics_v01, f, indent=4)
print("Metrics saved to metrics_v01.json")
