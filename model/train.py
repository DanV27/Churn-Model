import shap
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ---------- LOAD DATA ----------
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# keep only the 7 contract features + the target
df = df[['tenure', 'MonthlyCharges', 'Contract', 'InternetService',
         'PaymentMethod', 'OnlineSecurity', 'PaperlessBilling', 'Churn']]

# X = RAW features (not encoded — the pipeline does the encoding now)
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

# ---------- PREPROCESSOR ----------
numeric_features = ['tenure', 'MonthlyCharges']
categorical_features = ['Contract', 'InternetService', 'PaymentMethod',
                        'OnlineSecurity', 'PaperlessBilling']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# ---------- PIPELINE (preprocessor + model = ONE object) ----------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# ---------- SPLIT (on RAW X) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- TRAIN BASELINE ----------
pipeline.fit(X_train, y_train)                       # was model.fit(X_train_final, ...)
y_pred = pipeline.predict(X_test)                    # was model.predict(X_test_final)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("==================== MODEL EVALUATION — BASELINE ====================")
class_labels = ['NO CHURN', 'CHURN']
print(classification_report(y_test, y_pred, target_names=class_labels))

print("============================= METRICS ==============================")
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("======================= CONFUSION MATRIX ===========================")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in class_labels],
    columns=[f"Predicted {label}" for label in class_labels]
)
print(cm_df)

print("========================= CROSS-VALIDATION =========================")
scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')  # pipeline + raw X
print(f"Scores per fold: {scores}")
print(f"Mean ROC-AUC: {scores.mean():.3f}")
print(f"Standard Deviation: {scores.std():.3f}")

print("====================== HYPERPARAMETER TUNING =======================")
# keys are prefixed 'classifier__' because the model is nested in the pipeline
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10]
}
grid_search = GridSearchCV(
    estimator=pipeline,          # was estimator=model
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

print("===================== MODEL EVALUATION — TUNED =====================")
best_model = grid_search.best_estimator_   # this is a FULL pipeline, not just the RF
y_pred_tuned = best_model.predict(X_test)
y_proba_tuned = best_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_tuned, target_names=class_labels))

accuracy_tuned  = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned    = recall_score(y_test, y_pred_tuned)
f1_tuned        = f1_score(y_test, y_pred_tuned)
roc_auc_tuned   = roc_auc_score(y_test, y_proba_tuned)
print(f"Accuracy:  {accuracy_tuned:.4f}")
print(f"Precision: {precision_tuned:.4f}")
print(f"Recall:    {recall_tuned:.4f}")
print(f"F1-Score:  {f1_tuned:.4f}")
print(f"ROC-AUC:   {roc_auc_tuned:.4f}")

print("====================== SAVING MODEL =======================")
# ONE file now — the pipeline holds the encoder, scaler, AND model together
joblib.dump(best_model, "churn_model.pkl")
print("Saved full pipeline to 'churn_model.pkl'")
# (no more scaler.pkl — the scaler lives inside the pipeline)

print('=====================================================================')
# ---------- FEATURE IMPORTANCES ----------
# reach INTO the pipeline: model is under 'classifier',
# the encoded feature names come from the fitted 'preprocessor'
classifier = best_model.named_steps['classifier']
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances = classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)
print(len(importances), len(feature_names))



