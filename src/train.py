import argparse
import json
import joblib
import os
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def get_model(version: str):
    if version == "0.1":
        print("Using model: LinearRegression (v0.1)")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    elif version == "0.2":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                max_features="sqrt",
                random_state=RANDOM_SEED,
            )),
        ])
    else:
        raise ValueError(f"Unknown model version: {version}")
    return model

def main(version: str):
    print(f"--- Training model version {version} ---")

    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    pipeline = get_model(version)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Test RMSE for v{version}: {rmse:.4f}")

    os.makedirs("models", exist_ok=True)

    model_path = f"models/model_v{version}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    feature_path = "models/feature_list.json"
    with open(feature_path, "w") as f:
        json.dump(features, f)
    print(f"Features saved to {feature_path}")

    metrics = {"version": version, "rmse": rmse}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version to train (e.g., 0.1 or 0.2)"
    )
    args = parser.parse_args()
    main(args.version)
