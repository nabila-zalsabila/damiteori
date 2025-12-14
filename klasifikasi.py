import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD MODEL & SCALER
# =========================
svm_model = joblib.load("svm_model.pkl")
nb_model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit",
    layout="centered"
)

st.title("🩺 Aplikasi Prediksi Risiko Penyakit")
st.write("Menggunakan Machine Learning (SVM & Naive Bayes)")

# =========================
# PILIH MODEL
# =========================
model_choice = st.selectbox(
    "Pilih Model Klasifikasi",
    ("Support Vector Machine (SVM)", "Naive Bayes")
)

# =========================
# INPUT FITUR (CONTOH DATA HEART)
# SESUAIKAN DENGAN DATASET KAMU
# =========================
st.subheader("📝 Input Data Pasien")

age = st.number_input("Umur", min_value=1, max_value=120, value=50)
sex_m = st.selectbox("Jenis Kelamin", ("Perempuan", "Laki-laki"))
cp = st.selectbox("Chest Pain Type (0–3)", (0, 1, 2, 3))
trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 120)
chol = st.number_input("Kolesterol", 100, 600, 200)
thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)

# Encoding manual (HARUS SAMA dengan training)
sex_m = 1 if sex_m == "Laki-laki" else 0

# =========================
# BENTUK INPUT ARRAY
# URUTAN HARUS SAMA DENGAN X TRAINING
# =========================
input_data = np.array([[
    age,
    sex_m,
    cp,
    trestbps,
    chol,
    thalach
]])

# =========================
# TOMBOL PREDIKSI
# =========================
if st.button("🔍 Prediksi Risiko"):
    # Scaling
    input_scaled = scaler.transform(input_data)

    # Pilih model
    if model_choice == "Support Vector Machine (SVM)":
        model = svm_model
    else:
        model = nb_model

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    # Probabilitas (jika tersedia)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        prob_no = round(proba[0] * 100, 2)
        prob_yes = round(proba[1] * 100, 2)
    else:
        prob_no = prob_yes = None

    # =========================
    # OUTPUT HASIL
    # =========================
    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error("⚠️ Hasil: BERISIKO")
    else:
        st.success("✅ Hasil: TIDAK BERISIKO")

    if prob_no is not None:
        st.write(f"Probabilitas Tidak Berisiko: **{prob_no}%**")
        st.write(f"Probabilitas Berisiko: **{prob_yes}%**")

    st.info(f"Model yang digunakan: **{model_choice}**")
