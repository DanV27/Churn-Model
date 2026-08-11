import joblib
import pandas as pd

#Testing churn model on a test data set!

pipeline = joblib.load('model/churn_model.pkl')

customer = pd.DataFrame([{
    'tenure': 12,
    'MonthlyCharges': 79.90,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'PaymentMethod': 'Electronic check',
    'OnlineSecurity': 'No',
    'PaperlessBilling': 'Yes'
}])

prob = pipeline.predict_proba(customer)[:, 1][0]
print(f"Churn probability: {prob:.2%}")