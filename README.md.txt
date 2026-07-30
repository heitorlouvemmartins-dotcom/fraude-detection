# 🚨 Detecção de Fraudes em Transações

## 📌 Objetivo
Este projeto aplica técnicas de **Machine Learning** para detectar anomalias em transações financeiras, utilizando o dataset público de cartões de crédito.

## 🚀 Tecnologias
- Python 3.x
- Pandas, Numpy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib

## ⚙️ Funcionalidades
- Pré-processamento e engenharia de atributos (log e padronização)
- Treinamento de modelos de classificação:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Avaliação com métricas, matriz de confusão e curva ROC
- Comparação de desempenho via AUC
- Explicabilidade com SHAP e importância das variáveis

## 📊 Exemplo de Saída

===== Relatório XGBoost =====
precision    recall  f1-score   support

0       1.00      1.00      1.00     85307
1       0.92      0.85      0.88       135

accuracy                           1.00     85442
macro avg       0.96      0.92      0.94     85442
weighted avg       1.00      1.00      1.00     85442

Matriz de Confusão:
[[85295    12]
[   20   115]]

===== Resumo de AUC =====
Logistic Regression: 0.97
Random Forest: 0.99
XGBoost: 0.99


## 🏆 Aprendizados
- Importância do balanceamento em datasets desbalanceados
- Comparação entre modelos clássicos e avançados
- Uso de explicabilidade para entender variáveis críticas
- Documentação clara e reprodutibilidade para portfólio

## ▶️ Como executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/fraude-detection.git

pip install -r requirements.txt

python src/main.py

