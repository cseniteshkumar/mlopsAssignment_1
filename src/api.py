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

app = FastAPI(title="Assignment1 Model API")

MODEL_PATH = Path("models/model.joblib")
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(str(MODEL_PATH))
    else:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
