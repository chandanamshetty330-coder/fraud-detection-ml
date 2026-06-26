import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Credit Card Fraud Detection System")
st.markdown("Detect fraudulent transactions using XGBoost ML model trained on 284,807 real transactions.")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('../models/best_model.pkl')
    return model

model = load_model()

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app uses XGBoost ML model trained on 284,807 real credit card transactions.
- ROC AUC Score: 97.92%
- Top fraud indicators: V14, V4, V8
- Handles severe class imbalance using SMOTE
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost")
st.sidebar.markdown("**Dataset:** European Credit Card Transactions")
st.sidebar.markdown("**Built by:** Chandana")
st.sidebar.markdown("---")

# Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", [
    "🔍 Single Transaction",
    "📂 Batch Prediction",
    "📊 Model Performance",
    "📈 Data Insights"
])

# ==========================================
# Helper function to preprocess input
# ==========================================
def preprocess(v_features, amount, time):
    scaled_amount = (amount - 88.35) / 250.12
    scaled_time = (time - 94813) / 47488
    features = v_features + [scaled_amount, scaled_time]
    columns = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
    return pd.DataFrame([features], columns=columns)

# ==========================================
# PAGE 1 — Single Transaction
# ==========================================
if page == "🔍 Single Transaction":
    st.subheader("📝 Enter Transaction Details")

    col1, col2, col3 = st.columns(3)
    v_vals = []

    with col1:
        time = st.number_input("Time (seconds)", value=0.0)
        for i in range(1, 11):
            v_vals.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))

    with col2:
        for i in range(11, 21):
            v_vals.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))

    with col3:
        for i in range(21, 29):
            v_vals.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
        amount = st.number_input("Amount ($)", value=0.0, min_value=0.0)

    st.markdown("---")

    # Load sample transactions
    st.subheader("🧪 Quick Test")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info("🔴 Known Fraud: Set V1=-2.31, V2=1.95, V3=-1.61, V4=4.0, V14=-9.5, Amount=149.62")
    with col_t2:
        st.success("🟢 Known Legit: Leave all values at 0.0, Amount=10.0")

    st.markdown("---")

    if st.button("🔍 Predict Transaction", type="primary", use_container_width=True):
        input_df = preprocess(v_vals, amount, time)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        fraud_prob = float(probability[1] * 100)
        legit_prob = float(probability[0] * 100)

        st.markdown("## 🎯 Prediction Result")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            if prediction == 1:
                st.error("🚨 FRAUDULENT TRANSACTION")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")
        with col_r2:
            st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
        with col_r3:
            st.metric("Legitimate Probability", f"{legit_prob:.2f}%")

        st.markdown("### Fraud Risk Level")
        st.progress(float(fraud_prob / 100))

        if fraud_prob > 80:
            st.error("🔴 HIGH RISK — Block this transaction immediately!")
        elif fraud_prob > 50:
            st.warning("🟡 MEDIUM RISK — Flag for manual review")
        else:
            st.success("🟢 LOW RISK — Transaction appears legitimate")

        st.markdown("### 🔍 Key Features Analyzed")
        st.markdown(f"""
        - **V14** (strongest fraud indicator): `{v_vals[13]}`
        - **V4** (second strongest): `{v_vals[3]}`
        - **V8** (third strongest): `{v_vals[7]}`
        - **Amount**: `${amount}`
        """)

# ==========================================
# PAGE 2 — Batch Prediction
# ==========================================
elif page == "📂 Batch Prediction":
    st.subheader("📂 Batch Fraud Detection")
    st.markdown("Upload a CSV file with multiple transactions to detect fraud in bulk.")

    st.info("""
    **CSV Format Required:**
    - Columns: Time, V1 to V28, Amount
    - Each row = one transaction
    - Download sample below to see format
    """)

    # Sample CSV download
    sample_data = pd.DataFrame({
        'Time': [0, 1000, 2000],
        **{f'V{i}': [0.0, 0.0, 0.0] for i in range(1, 29)},
        'Amount': [149.62, 2.69, 378.66]
    })

    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_buffer.getvalue(),
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"✅ Loaded {len(df)} transactions")
        st.dataframe(df.head())

        if st.button("🔍 Detect Fraud in All Transactions", type="primary"):
            with st.spinner("Analyzing transactions..."):
                # Preprocess
                df['scaled_amount'] = (df['Amount'] - 88.35) / 250.12
                df['scaled_time'] = (df['Time'] - 94813) / 47488
                df_model = df.drop(['Time', 'Amount'], axis=1)

                # Predict
                predictions = model.predict(df_model)
                probabilities = model.predict_proba(df_model)[:, 1]

                # Results
                df['Prediction'] = predictions
                df['Fraud_Probability'] = (probabilities * 100).round(2)
                df['Status'] = df['Prediction'].map({0: '✅ Legitimate', 1: '🚨 Fraud'})

                # Summary
                total = len(df)
                fraud_count = df['Prediction'].sum()
                legit_count = total - fraud_count

                st.markdown("## 📊 Batch Results Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", total)
                with col2:
                    st.metric("Fraudulent", int(fraud_count), delta=f"{fraud_count/total*100:.1f}%")
                with col3:
                    st.metric("Legitimate", int(legit_count))

                # Pie chart
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(
                    [legit_count, fraud_count],
                    labels=['Legitimate', 'Fraud'],
                    colors=['#2ecc71', '#e74c3c'],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.set_title('Transaction Distribution')
                st.pyplot(fig)
                plt.close()

                # Show fraud transactions
                st.markdown("### 🚨 Flagged Fraud Transactions")
                fraud_df = df[df['Prediction'] == 1][['Amount', 'Fraud_Probability', 'Status']]
                st.dataframe(fraud_df)

                # Download results
                result_buffer = io.StringIO()
                df[['Amount', 'Fraud_Probability', 'Status']].to_csv(result_buffer, index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=result_buffer.getvalue(),
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

# ==========================================
# PAGE 3 — Model Performance
# ==========================================
elif page == "📊 Model Performance":
    st.subheader("📊 Model Performance Metrics")

    # Model comparison
    st.markdown("### 🏆 Model Comparison")
    model_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
        'ROC AUC': [0.9698, 0.9688, 0.9792, 0.9694],
        'Status': ['Good', 'Good', '🏆 Best', 'Good']
    })
    st.dataframe(model_data, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#3498db', '#3498db', '#e74c3c', '#3498db']
    bars = ax.bar(model_data['Model'], model_data['ROC AUC'], color=colors)
    ax.set_ylim(0.96, 0.985)
    ax.set_title('ROC AUC Score Comparison')
    ax.set_ylabel('ROC AUC Score')
    for bar, val in zip(bars, model_data['ROC AUC']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Why XGBoost won
    st.markdown("### 🤔 Why XGBoost Won")
    st.markdown("""
    - Handles **imbalanced data** better than Logistic Regression
    - Uses **gradient boosting** — learns from previous mistakes
    - Built-in **regularization** prevents overfitting
    - Faster than Random Forest with better accuracy
    - Industry standard for **tabular data** problems
    """)

    st.markdown("---")

    # SHAP insights
    st.markdown("### 🔍 Top Features (SHAP Analysis)")
    shap_data = pd.DataFrame({
        'Feature': ['V14', 'V4', 'V8', 'V12', 'V18', 'V1', 'V22', 'V11', 'V24', 'V3'],
        'SHAP Importance': [2.32, 1.98, 0.79, 0.78, 0.77, 0.76, 0.69, 0.68, 0.60, 0.60]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_data['Feature'], shap_data['SHAP Importance'], color='crimson')
    ax.set_xlabel('Mean SHAP Value')
    ax.set_title('Top 10 Features Driving Fraud Detection')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ==========================================
# PAGE 4 — Data Insights
# ==========================================
elif page == "📈 Data Insights":
    st.subheader("📈 Dataset Insights")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "284,807")
    with col2:
        st.metric("Fraud Cases", "492")
    with col3:
        st.metric("Fraud Rate", "0.17%")
    with col4:
        st.metric("Features", "30")

    st.markdown("---")

    # Class imbalance chart
    st.markdown("### ⚖️ Class Imbalance")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [284315, 492],
        labels=['Legitimate (99.83%)', 'Fraud (0.17%)'],
        colors=['#2ecc71', '#e74c3c'],
        autopct='%1.2f%%',
        startangle=90
    )
    ax.set_title('Transaction Distribution')
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Key findings
    st.markdown("### 🔑 Key EDA Findings")
    st.markdown("""
    - **Severe class imbalance** — only 0.17% fraud cases → solved using SMOTE
    - **Fraud amounts** are generally lower than legitimate transactions
    - **V14 and V4** are the strongest predictors of fraud
    - **Time** plays a role — fraud patterns differ across time periods
    - **No missing values** in the dataset
    - All features **V1-V28 are PCA transformed** for privacy protection
    """)

    st.markdown("---")

    # Tech stack
    st.markdown("### 🛠️ Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Data Processing:**
        - Python 3.10
        - Pandas
        - NumPy
        - Scikit-learn
        """)
    with col2:
        st.markdown("""
        **ML & Visualization:**
        - XGBoost
        - SHAP
        - Matplotlib
        - Streamlit
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using XGBoost + Streamlit | Credit Card Fraud Detection Project | Chandana")