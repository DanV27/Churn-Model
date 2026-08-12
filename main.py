from typing import Literal
import joblib
import shap
import pandas as pd
from fastapi import FastAPI, status
from pydantic import BaseModel

pipeline = joblib.load('model/churn_model.pkl')

# pull the fitted pieces straight out of the saved pipeline — no retraining
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']
explainer = shap.TreeExplainer(classifier)
feature_names = preprocessor.get_feature_names_out()

# 1. Request schema — the 7 input features (NO Churn, that's what we predict)
class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    PaperlessBilling: Literal["Yes", "No"]

# 2. Response schema — what the caller gets back


class Factor(BaseModel):
    feature: str
    impact: float
    direction: str

class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    top_factors: list[Factor]

app = FastAPI()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {"message": "Welcome! Go to /health for the health check."}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    row = pd.DataFrame([customer.model_dump()])

    # probability (same as before)
    probability = pipeline.predict_proba(row)[:, 1][0]

    # --- SHAP for this one customer ---
    row_transformed = preprocessor.transform(row)
    shap_values = explainer(row_transformed)

    # one customer, all features, churn class (class 1)
    customer_shap = shap_values[0, :, 1].values   # numpy array, one value per encoded column

    # group encoded columns back to original features
    impacts = {}
    for name, val in zip(feature_names, customer_shap):
        # 'num__tenure' -> 'tenure';  'cat__Contract_Two year' -> 'Contract'
        original = name.split('__')[1].split('_')[0]
        impacts[original] = impacts.get(original, 0) + val

    # top 3 by absolute impact
    top = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_factors = [
        {
            "feature": feat,
            "impact": round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk"
        }
        for feat, val in top
    ]

    return {
        "churn_probability": float(probability),
        "will_churn": bool(probability >= 0.5),
        "top_factors": top_factors
    }