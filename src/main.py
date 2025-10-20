import os
import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# --- 配置 ---
# 关键：从环境变量读取要加载哪个版本的模型
# Dockerfile 会在构建时设置这个变量
MODEL_VERSION = os.environ.get("MODEL_VERSION", "0.1")
MODEL_PATH = f"models/model_v{MODEL_VERSION}.joblib"
FEATURES_PATH = "models/feature_list.json"

# --- Pydantic 模型 (数据验证) ---
class PatientFeatures(BaseModel):
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
                "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
                "s5": 0.02, "s6": -0.001
            }
        }

class PredictionResponse(BaseModel):
    prediction: float

class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str

# --- FastAPI 应用 ---
app = FastAPI(title="Diabetes Clinic API", version="1.0")

@app.on_event("startup")
def load_artifacts():
    """在 API 启动时加载模型和特征列表"""
    try:
        app.state.model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r") as f:
            app.state.features = json.load(f)
        print(f"--- Model v{MODEL_VERSION} and features loaded successfully ---")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        app.state.model = None
        app.state.features = None

# --- API 接口 ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查接口，返回当前运行的模型版本"""
    if app.state.model:
        return {"status": "ok", "model_version": MODEL_VERSION}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict", response_model=PredictionResponse)
def predict(patient_data: PatientFeatures):
    """预测接口"""
    if not app.state.model or not app.state.features:
        raise HTTPException(status_code=503, detail="Model or features not loaded")

    try:
        data_df = pd.DataFrame([patient_data.model_dump()])
        data_df = data_df[app.state.features] 
        prediction = app.state.model.predict(data_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
