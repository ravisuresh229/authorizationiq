import pickle
import pandas as pd
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import threading
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 config
s3 = boto3.client('s3')
BUCKET_NAME = 'pa-predictor-bucket-rs'
MODEL_FILE = 'pa_predictor_model.pkl'
CHECK_INTERVAL = 60

# Globals
model = None
last_modified = None

# Load valid codes for validation
def load_valid_codes():
    cpt = pd.read_csv('cpt_codes.csv')
    icd = pd.read_csv('icd10_codes.csv')
    specialties = pd.read_csv('provider_specialties.csv')
    return (
        set(cpt['code'].str.strip().str.upper()),
        set(icd['code'].str.strip().str.upper()),
        set(specialties['specialty'].dropna())
    )

valid_cpt_codes, valid_icd10_codes, valid_specialties = load_valid_codes()

# Load model
def load_model_from_s3():
    global model, last_modified
    try:
        logger.info("🔍 Checking model version in S3...")
        response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_FILE)
        new_modified = response['LastModified']
        if new_modified != last_modified:
            last_modified = new_modified
            model = pickle.loads(response['Body'].read())
            logger.info("✅ Model reloaded from S3.")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")

threading.Thread(target=lambda: (load_model_from_s3(), time.sleep(CHECK_INTERVAL)), daemon=True).start()
load_model_from_s3()

# FastAPI app
app = FastAPI(
    title="PA Predictor API",
    description="Predict prior authorization approval",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "PA Predictor API is live"}

@app.get("/health")
def health_check():
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# Input model
class PredictionInput(BaseModel):
    patient_age: int = Field(..., ge=18, le=90)
    patient_gender: Literal["M", "F"]
    procedure_code: str
    diagnosis_code: str
    provider_specialty: str
    payer: str
    urgency_flag: Literal["Y", "N"]
    documentation_complete: Literal["Y", "N"]
    prior_denials_provider: int = Field(..., ge=0, le=10)
    region: Literal["Midwest", "Northeast", "South", "West"]

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([input_data.dict()])

    try:
        pred = model.predict(df)[0]
        prob = float(model.predict_proba(df)[0][1])
        return {
            "approval_prediction": int(pred),
            "probability": round(prob, 4),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check input format.") 