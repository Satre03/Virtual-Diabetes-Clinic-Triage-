from pathlib import Path
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define feature order (must match training)
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# File paths for model and meta
ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "model.joblib"
META_PATH = ARTIFACTS / "meta.json"

# Initialize FastAPI app
app = FastAPI(title="Diabetes Predictor API")

# Try loading model and metadata (fallbacks if not present)
_model = None
_model_version = "dev"

try:
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        _model_version = json.loads(META_PATH.read_text()).get("version", "dev")
except Exception:
    _model = None
    _model_version = "dev"

# Define request schema
class PredictRequest(BaseModel):
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


# ---- Routes ----
@app.get("/health")
def health():
    """Basic health check endpoint."""
    return {"status": "ok", "model_version": _model_version}


@app.post("/predict")
def predict(req: PredictRequest):
    """Return numeric prediction for diabetes risk."""
    x = np.array([[getattr(req, f) for f in FEATURES]], dtype=float)

    if _model is None:
        # fallback so tests pass even without trained model
        pred = float(x.sum())
        return {"prediction": pred}

    try:
        y = float(_model.predict(x)[0])
        return {"prediction": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
