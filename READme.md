# 🔍 Credit Card Fraud Detection System

A complete end-to-end Machine Learning project that detects fraudulent credit card transactions using XGBoost with 97.92% ROC AUC score.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-green)
![ROC AUC](https://img.shields.io/badge/ROC%20AUC-97.92%25-brightgreen)

---

## 🎯 Problem Statement

Credit card fraud causes billions in losses every year. This project builds an intelligent system that automatically detects fraudulent transactions in real-time, helping banks and fintech companies protect their customers.

---

## 🚀 Live Demo

👉 **[Click here to try the live app](https://fraud-detection-ml-pu7y.onrender.com)**← (add Render link after deployment)

---

## 📊 Project Highlights

- ✅ Trained on **284,807 real credit card transactions**
- ✅ Handled severe **class imbalance (0.17% fraud)** using SMOTE
- ✅ Compared **4 ML models** — XGBoost won with **97.92% ROC AUC**
- ✅ **SHAP explainability** — identified V14 and V4 as top fraud indicators
- ✅ **Live Streamlit app** with single + batch prediction
- ✅ **Batch CSV upload** — detect fraud in thousands of transactions at once

---

## 🖥️ App Features

| Feature | Description |
|---------|-------------|
| 🔍 Single Transaction | Enter transaction details → get instant fraud probability |
| 📂 Batch Prediction | Upload CSV → detect fraud in all transactions → download results |
| 📊 Model Performance | Compare all 4 models, ROC AUC scores, SHAP feature importance |
| 📈 Data Insights | Dataset statistics, class imbalance, EDA findings |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Render |

---

## 📈 Model Comparison

| Model | ROC AUC |
|-------|---------|
| Logistic Regression | 0.9698 |
| Random Forest | 0.9688 |
| **XGBoost** | **0.9792 🏆** |
| LightGBM | 0.9694 |

---

## 🔍 SHAP Feature Importance

Top features driving fraud detection:
1. **V14** — SHAP: 2.32 (strongest fraud indicator)
2. **V4** — SHAP: 1.98
3. **V8** — SHAP: 0.79
4. **V12** — SHAP: 0.78
5. **V18** — SHAP: 0.77

---

## 📁 Project Structure

fraud-detection-ml/

├── data/

│   └── creditcard.csv

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_modelling.ipynb

│   └── 03_shap.ipynb

├── src/

├── app/

│   └── app.py

├── models/

│   └── best_model.pkl

├── requirements.txt

└── README.md

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
cd app
streamlit run app.py
```

---

## 📊 Key Findings from EDA

- Severe class imbalance — only **0.17% fraud cases** out of 284,807 transactions
- Fraud transactions have **lower average amounts** than legitimate ones
- **V14 and V4** are the strongest predictors of fraud
- Time plays a role — fraud patterns differ across time periods
- No missing values in the dataset

---

## 🧠 What I Learned

- Handling severely imbalanced datasets using SMOTE
- Why ROC AUC is better than accuracy for imbalanced data
- How SHAP explains black-box ML models to business stakeholders
- Building and deploying production-ready ML applications

---

## 👩‍💻 Built By

**Chandana M** — Data Science Student  
📧 shettychandanam@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/chandana-m-b5966b368)  
🐱 [GitHub](https://github.com/chandanamshetty330-coder)

---

## 📄 Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
