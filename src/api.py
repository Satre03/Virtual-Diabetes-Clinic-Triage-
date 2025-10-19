from pathlib import Path
from contextlib import asynccontextmanager
import os
#import json
import joblib
import numpy as np  # required later, used for dummy model / array conversion

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ART_DIR = Path("artifacts")


class InputFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None


def load_model():
    """Load trained model from artifacts folder."""
    mdl_path = ART_DIR / "model.joblib"
    if not mdl_path.exists():
        # Allow dummy model for startup in CI smoke test
        if os.getenv("ALLOW_DUMMY_MODEL", "1") == "1":
            print("⚠️ No model found — using dummy model.")
            return lambda X: np.array([0.0])
        raise HTTPException(status_code=503, detail="Model not found.")
    print(f"✅ Loaded model from {mdl_path}")
    return joblib.load(mdl_path)


app = FastAPI(title="Virtual Diabetes Clinic Triage", lifespan=lifespan)


@app.get("/health")
def health():
    model_version = os.getenv("MODEL_VERSION", "dev")
    return {"status": "ok", "model_version": model_version}


@app.post("/predict")
def predict(features: InputFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        vec = [
            features.age,
            features.sex,
            features.bmi,
            features.bp,
            features.s1,
            features.s2,
            features.s3,
            features.s4,
            features.s5,
            features.s6,
        ]
        X = np.asarray(vec, dtype=float).reshape(1, -1)
        y = float(model.predict(X)[0])  # sklearn-like predict
        return {"prediction": y}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
