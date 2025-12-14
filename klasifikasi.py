import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD MODEL, SCALER, FITUR
# =========================
svm_model = joblib.load("svm_model.pkl")
nb_model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

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
    ["Support Vector Machine (SVM)", "Naive Bayes"]
)

# =========================
# INPUT DATA PASIEN
# =========================
st.subheader("📝 Input Data Pasien")

age = st.number_input("Umur", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 120)
chol = st.number_input("Kolesterol", 100, 600, 200)
thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)

# =========================
# BENTUK INPUT SESUAI TRAINING
# =========================
input_df = pd.DataFrame(
    np.zeros((1, len(feature_columns))),
    columns=feature_columns
)

# Isi fitur numerik
if "age" in input_df.columns:
    input_df["age"] = age

if "trestbps" in input_df.columns:
    input_df["trestbps"] = trestbps

if "chol" in input_df.columns:
    input_df["chol"] = chol

if "thalach" in input_df.columns:
    input_df["thalach"] = thalach

# Encoding gender
if "Sex_M" in input_df.columns:
    input_df["Sex_M"] = 1 if sex == "Laki-laki" else 0

# Encoding chest pain (jika one-hot)
for i in range(4):
    col_name = f"ChestPainType_{i}"
    if col_name in input_df.columns:
        input_df[col_name] = 1 if cp == i else 0

# =========================
# TOMBOL PREDIKSI
# =========================
if st.button("🔍 Prediksi Risiko"):
    # Scaling (ANTI ERROR)
    input_scaled = scaler.transform(input_df)

    # Pilih model
    model = svm_model if model_choice == "Support Vector Machine (SVM)" else nb_model

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

    # Gunakan PROBABILITAS, bukan label keras
    if prob_yes >= 70:
        st.error("⚠️ RISIKO TINGGI")
    elif prob_yes >= 50:
        st.warning("⚠️ RISIKO SEDANG")
    else:
        st.success("✅ RISIKO RENDAH")

    st.write(f"Probabilitas Tidak Berisiko: **{prob_no}%**")
    st.write(f"Probabilitas Berisiko: **{prob_yes}%**")

    st.info(f"Model yang digunakan: **{model_choice}**")

    st.caption(
        "⚠️ Hasil prediksi bersifat estimasi berbasis data dan tidak menggantikan diagnosis dokter."
    )

# =========================
# DEBUG OPSIONAL (HAPUS JIKA TIDAK PERLU)
# =========================
with st.expander("🔧 Debug (opsional)"):
    st.write("Jumlah fitur input:", input_df.shape[1])
    st.write("Jumlah fitur scaler:", scaler.n_features_in_)
