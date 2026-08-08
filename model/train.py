
#Imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
#read csv file
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
#dropping customer since it is not needed
df = df.drop(columns=['customerID']) #DROPS customerID
#Convert the actual DataFrame column and save it back
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#Drop rows if total charges is NaN
df = df.dropna(subset=['TotalCharges'])
# X takes everything EXCEPT the last column
X = df.iloc[:, :-1] #Features
# y takes ONLY the last column
y = df.iloc[:, -1] #Churn(target)
#turning churn yes/no values into 1/0 integers
y = y.map({'Yes': 1, 'No': 0})
# Standard encoding that automatically finds categorical columns
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Convert the scaled arrays back into pandas DataFrames
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
#Export to CSV files (index=False drops the row numbers) and adds target variable to final column
X_train_final.assign(Churn=y_train).to_csv('X_train_scaled.csv', index=False)
X_test_final.assign(Churn=y_test).to_csv('X_test_scaled.csv', index=False)
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_final, y_train)
#both predict() AND predict_prob
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)[:, 1]
print("==================== MODEL EVALUATION — BASELINE ====================")
#labels
class_labels = ['NO CHURN', 'CHURN']
print(classification_report(y_test, y_pred, target_names=class_labels))
print("============================= METRICS ==============================")
#calculate metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)
#print metrics
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("======================= CONFUSION MATRIX ===========================")
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in class_labels],
    columns=[f"Predicted {label}" for label in class_labels]
)
print(cm_df)
scores = cross_val_score(model, X_encoded, y, cv=5, scoring='roc_auc')
print("========================= CROSS-VALIDATION =========================")
print(f"Scores per fold: {scores}")
print(f"Mean ROC-AUC: {scores.mean():.3f}")
print(f"Standard Deviation: {scores.std():.3f}")
print("====================== HYPERPARAMETER TUNING =======================")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',       # Options: 'accuracy', 'precision', 'recall', 'roc_auc', etc.
    cv=5,
    n_jobs=-1           # Uses all available CPU cores to speed it up
)
grid_search.fit(X_train_final, y_train)
# Print the best combination of parameters found
print("Best Parameters:", grid_search.best_params_)
# Print the best cross-validation score achieved during training
print("Best CV Score:", grid_search.best_score_)
print("===================== MODEL EVALUATION — TUNED =====================")
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_final)
y_proba_tuned = best_model.predict_proba(X_test_final)[:, 1]
print(classification_report(y_test, y_pred_tuned, target_names=class_labels))
#calculate metrics
accuracy_tuned  = accuracy_score(y_test, y_pred_tuned)
precision_tuned  = precision_score(y_test, y_pred_tuned)
recall_tuned     = recall_score(y_test, y_pred_tuned)
f1_tuned         = f1_score(y_test, y_pred_tuned)
roc_auc_tuned    = roc_auc_score(y_test, y_proba_tuned)
#print metrics
print(f"Accuracy:  {accuracy_tuned :.4f}")
print(f"Precision: {precision_tuned :.4f}")
print(f"Recall:    {recall_tuned :.4f}")
print(f"F1-Score:  {f1_tuned :.4f}")
print(f"ROC-AUC:   {roc_auc_tuned :.4f}")