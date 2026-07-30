# main.py - Projeto de Detecção de Fraudes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier

# 1. Carregar dados
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# 2. Feature Engineering
df["amount_log"] = np.log1p(df["Amount"])  # log1p evita problemas com valores zero
scaler = StandardScaler()
df["amount_scaled"] = scaler.fit_transform(df[["amount_log"]])

X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Função de relatório
def relatorio_modelo(y_true, y_pred, nome_modelo):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n===== Relatório {nome_modelo} =====")
    print(classification_report(y_true, y_pred))
    print("Matriz de Confusão:\n", cm)

# 4. Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
relatorio_modelo(y_test, y_pred_log, "Logistic Regression")

# ROC Curve
y_probs_log = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs_log)
plt.plot(fpr, tpr)
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
print("AUC Logistic Regression:", roc_auc_score(y_test, y_probs_log))

# 5. Random Forest
rf = RandomForestClassifier(
    n_estimators=50, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
relatorio_modelo(y_test, y_pred_rf, "Random Forest")
y_probs_rf = rf.predict_proba(X_test)[:, 1]

# 6. XGBoost
xgb = XGBClassifier(scale_pos_weight=10, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
relatorio_modelo(y_test, y_pred_xgb, "XGBoost")
y_probs_xgb = xgb.predict_proba(X_test)[:, 1]

# Importância das variáveis com labels
importancias = xgb.feature_importances_
plt.figure(figsize=(12,6))
plt.bar(range(len(importancias)), importancias)
plt.xticks(range(len(importancias)), X.columns, rotation=90)
plt.title("Importância das Variáveis - XGBoost")
plt.show()

# 7. Explicabilidade com SHAP
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100])  # limitar para performance
shap.plots.bar(shap_values)

# 8. Resumo comparativo de AUC
print("\n===== Resumo de AUC =====")
print("Logistic Regression:", roc_auc_score(y_test, y_probs_log))
print("Random Forest:", roc_auc_score(y_test, y_probs_rf))
print("XGBoost:", roc_auc_score(y_test, y_probs_xgb))