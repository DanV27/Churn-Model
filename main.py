import joblib
import pandas as pd
from fastapi import FastAPI, status
from pydantic import BaseModel

pipeline = joblib.load('model/churn_model.pkl')

# 1. Request schema — the 7 input features (NO Churn, that's what we predict)
class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    OnlineSecurity: str
    PaperlessBilling: str

# 2. Response schema — what the caller gets back
class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool

app = FastAPI()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {"message": "Welcome! Go to /health for the health check."}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    row = pd.DataFrame([{
        'tenure': customer.tenure,
        'MonthlyCharges': customer.MonthlyCharges,
        'Contract': customer.Contract,
        'InternetService': customer.InternetService,
        'PaymentMethod': customer.PaymentMethod,
        'OnlineSecurity': customer.OnlineSecurity,
        'PaperlessBilling': customer.PaperlessBilling
    }])

    probability = pipeline.predict_proba(row)[:, 1][0]

    return {
        "churn_probability": float(probability),
        "will_churn": bool(probability >= 0.5)
    }