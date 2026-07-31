import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Klasifikasi Kanker", layout="wide")

st.title("🧬 Aplikasi Klasifikasi Kanker dengan Machine Learning")

# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset")
    st.write(df.head())

    # =========================
    # PILIH TARGET
    # =========================
    target = st.selectbox("Pilih Kolom Target", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    # =========================
    # ENCODING
    # =========================
    X = pd.get_dummies(X)

    # =========================
    # SPLIT DATA
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # PILIH MODEL
    # =========================
    st.sidebar.title("⚙️ Pilih Model")
    model_name = st.sidebar.selectbox(
        "Model",
        ["Random Forest", "SVM", "Logistic Regression"]
    )

    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "SVM":
        model = SVC()
    else:
        model = LogisticRegression(max_iter=1000)

    # =========================
    # TRAIN BUTTON
    # =========================
    if st.button("🚀 Train Model"):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # =========================
        # METRIK
        # =========================
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.subheader("📈 Hasil Evaluasi")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("Precision", f"{prec:.2f}")
        col3.metric("Recall", f"{rec:.2f}")
        col4.metric("F1-Score", f"{f1:.2f}")

        # =========================
        # TABEL
        # =========================
        results = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [acc, prec, rec, f1]
        })

        st.subheader("📋 Tabel Evaluasi")
        st.dataframe(results)

        # =========================
        # GRAFIK
        # =========================
        st.subheader("📊 Grafik Evaluasi")
        fig, ax = plt.subplots()
        ax.bar(results['Metric'], results['Value'])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # =========================
        # CONFUSION MATRIX
        # =========================
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2)
        st.pyplot(fig2)

        # =========================
        # PREDIKSI MANUAL
        # =========================
        st.subheader("🔮 Prediksi Data Baru")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Prediksi"):
            input_df = pd.DataFrame([input_data])
            input_df = scaler.transform(input_df)
            prediction = model.predict(input_df)

            st.success(f"Hasil Prediksi: {prediction[0]}")