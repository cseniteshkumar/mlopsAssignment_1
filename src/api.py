# ...existing code...
"""FastAPI server wrapping the project's prediction pipeline.

Provides:
 - GET /health
 - POST /predict (CSV file upload)

The server loads `models/model.joblib` at startup. If the model is missing, /health
will indicate it and /predict will return 503.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from io import BytesIO
import joblib
import pandas as pd
import traceback
from typing import List, Dict, Optional, Union

# ...existing code...

app = FastAPI(title="Assignment1 Model API")

# Resolve model path relative to this file so app works regardless of CWD
def _find_model_path() -> Path:
    base = Path(__file__).resolve().parent.parent  # project root
    candidates = [
        base / "models" / "deployed_model.joblib",
        base / "models" / "model.joblib",
        base / "models" / "model.pkl",
        base / "model.joblib",
    ]
    for p in candidates:
        if p.exists():
            return p
    # default to common location even if not present
    return base / "models" / "deployed_model.joblib"

MODEL_PATH = _find_model_path()
model = None
model_load_error: Optional[str] = None

@app.on_event("startup")
def load_model():
    global model, model_load_error, MODEL_PATH
    try:
        if MODEL_PATH.exists():
            model = joblib.load(str(MODEL_PATH))
            model_load_error = None
            print(f"Loaded model from {MODEL_PATH}")
        else:
            model = None
            model_load_error = f"Model path does not exist: {MODEL_PATH}"
            print(model_load_error)
    except Exception as e:
        model = None
        model_load_error = f"Failed to load model: {e}"
        print(model_load_error)
        traceback.print_exc()

# ...existing code...
@app.get("/")
def read_root():
    return {"status": "Server is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "load_error": model_load_error,
    }

# ...existing code...
@app.post("/predictFile")
async def predictFile(file: UploadFile = File(...)):
    """
    Predict from an uploaded CSV file (multiple rows allowed).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        # Drop common target column names if present
        for c in ["target", "label", "class", "y", "num"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        preds = model.predict(df)
        return {"predictions": [int(p) if (hasattr(p, "__int__")) else p for p in preds.tolist()]}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Failed to predict: {e}\n{tb}")


@app.post("/predict")
async def predict(features: dict | None = None):
    """
    Predict for a single sample passed as JSON in the request body.
    Example JSON:
      { "features": {"feat1": 1.2, "feat2": 3, "feat3": "A"} }

    Accepts:
      - features as a dict (mapping feature_name -> value)
      - if the top-level JSON is a dict of features directly (FastAPI will send it to `features` param),
        it will work as well.

    Returns a single prediction.

    Sample Call :
 \n
   {
    "age": 63.0,
    "sex": 1.0,
    "cp": 1.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 2.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 3.0,
    "ca": 0.0,
    "thal": 6.0 }

    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if features is None:
        raise HTTPException(status_code=400, detail="Missing 'features' JSON payload")

    try:
        # If features is a mapping, create single-row DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            # list of values -> single-row with numeric positions; user must ensure correct order
            df = pd.DataFrame([features])
        else:
            raise HTTPException(status_code=400, detail="Unsupported features payload type")

        # Drop common target column names if present
        for c in ["target", "label", "class", "y", "num"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        preds = model.predict(df)
        p = preds[0]
        return {"prediction": int(p) if (hasattr(p, "__int__")) else p}
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Failed to predict single sample: {e}\n{tb}")
