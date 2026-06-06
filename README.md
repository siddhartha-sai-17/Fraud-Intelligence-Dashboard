# 🛡️ Fraud Intelligence Dashboard

## 📌 Overview
**Fraud Intelligence Dashboard** is an AI-powered deep learning system designed to detect fraudulent financial transactions using sequential modeling techniques such as **LSTM**, **Attention Mechanism**, and **Positional Encoding**.

It provides:
- Real-time fraud detection
- Batch CSV analysis
- Explainable AI visualizations
- Interactive Streamlit dashboard

---

## 🚀 Features

- 📊 CSV-based batch fraud prediction  
- ⚡ Real-time transaction simulation  
- 🧠 Deep learning models (Dense, LSTM, Attention, PE + Attention)  
- 🔍 Explainable AI using attention visualization  
- 🚨 High-risk transaction detection system  
- 📈 Model comparison dashboard (Accuracy, Precision, Recall, F1, ROC-AUC)  
- 🎯 Fraud probability scoring and risk classification  

---

## 🏗️ Project Architecture

```text
Transaction Sequence (Last 4 Transactions)
        ↓
Feature Scaling (StandardScaler)
        ↓
Deep Learning Model (LSTM / Attention / PE)
        ↓
Fraud Probability Prediction
        ↓
Risk Classification Engine
        ↓
Streamlit Dashboard Visualization