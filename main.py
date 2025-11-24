"""
Simple FastAPI app to serve predictions.
POST /predict with JSON body: {"Length1":..., "Length2":..., "Length3":..., "Height":..., "Width":..., "Species": "Bream"}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import List

MODEL_PATH = os.environ.get("FISH_MODEL_PATH", "models/fish_model.joblib")

app = FastAPI(title="Fish Weight Predictor")

pipeline = None


class FishSample(BaseModel):
    Length1: float
    Length2: float
    Length3: float
    Height: float
    Width: float
    Species: str


@app.on_event("startup")
def load_model():
    global pipeline
    if not os.path.exists(MODEL_PATH):
        # do not crash here in case user wants to run docs without model
        pipeline = None
        return
    pipeline = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(sample: FishSample):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not available. Train the model and set FISH_MODEL_PATH or place models/fish_model.joblib")
    try:
        preprocessor = pipeline["preprocessor"]
        model = pipeline["model"]
        X = [[sample.Length1, sample.Length2, sample.Length3, sample.Height, sample.Width, sample.Species]]
        X_trans = preprocessor.transform(X)
        pred = model.predict(X_trans)
        return {"predicted_weight": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
def batch_predict(samples: List[FishSample]):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not available.")
    try:
        rows = [[s.Length1, s.Length2, s.Length3, s.Height, s.Width, s.Species] for s in samples]
        X_trans = pipeline["preprocessor"].transform(rows)
        preds = pipeline["model"].predict(X_trans)
        return {"predicted_weights": [float(v) for v in preds]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
