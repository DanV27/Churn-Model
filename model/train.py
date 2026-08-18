import json
import shap
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# paths are anchored to this file, so the script runs from any working directory
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = MODEL_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_PATH = MODEL_DIR / 'churn_model.pkl'
THRESHOLD_PATH = MODEL_DIR / 'threshold.json'

# A missed churner costs far more than a needless retention offer: you lose the
# account, they get a discount they might not have needed. 5:1 encodes that
# asymmetry — change these two numbers and the chosen threshold moves with them.
FALSE_NEGATIVE_COST = 5.0   # customer churns and we did nothing
FALSE_POSITIVE_COST = 1.0   # retention offer sent to someone who would have stayed

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)

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


print("=================== DECISION THRESHOLD SELECTION ===================")
# predict() hardcodes 0.5, which silently assumes both mistakes cost the same.
# They don't, so pick the threshold that minimises expected cost instead.
thresholds = np.arange(0.01, 1.00, 0.01)
rows = []
for t in thresholds:
    predicted = (y_proba_tuned >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    rows.append({
        'threshold': round(float(t), 2),
        'cost': fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST,
        'precision': precision_score(y_test, predicted, zero_division=0),
        'recall': recall_score(y_test, predicted, zero_division=0),
        'f1': f1_score(y_test, predicted, zero_division=0),
        'flagged': int(tp + fp),
    })

sweep_df = pd.DataFrame(rows)
best_row = sweep_df.loc[sweep_df['cost'].idxmin()]
best_threshold = float(best_row['threshold'])
default_row = sweep_df[sweep_df['threshold'] == 0.50].iloc[0]

print(f"Cost model: false negative = {FALSE_NEGATIVE_COST}, false positive = {FALSE_POSITIVE_COST}")
print(sweep_df[sweep_df['threshold'].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]
      .to_string(index=False))
print()
print(f"Default 0.50  -> cost {default_row['cost']:.0f}, "
      f"precision {default_row['precision']:.4f}, recall {default_row['recall']:.4f}, "
      f"flagged {default_row['flagged']}")
print(f"Chosen {best_threshold:.2f}  -> cost {best_row['cost']:.0f}, "
      f"precision {best_row['precision']:.4f}, recall {best_row['recall']:.4f}, "
      f"flagged {int(best_row['flagged'])}")
print(f"Expected cost reduction: {(1 - best_row['cost'] / default_row['cost']) * 100:.1f}%")

THRESHOLD_PATH.write_text(json.dumps({
    'threshold': best_threshold,
    'false_negative_cost': FALSE_NEGATIVE_COST,
    'false_positive_cost': FALSE_POSITIVE_COST,
    'test_set_precision': round(float(best_row['precision']), 4),
    'test_set_recall': round(float(best_row['recall']), 4),
    'test_set_f1': round(float(best_row['f1']), 4),
}, indent=2) + "\n")
print(f"Wrote {THRESHOLD_PATH}")

print("====================== SAVING MODEL =======================")
# ONE file now — the pipeline holds the encoder, scaler, AND model together
joblib.dump(best_model, MODEL_PATH)
print(f"Saved full pipeline to '{MODEL_PATH}'")
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



