import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="centered"
)

st.title("❤️ Aplikasi Prediksi Penyakit Jantung")
st.write("Menggunakan Machine Learning (Logistic Regression)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

st.subheader("📊 Dataset")
st.dataframe(df.head())

# =========================
# TENTUKAN TARGET
# =========================
possible_targets = ["target", "output", "HeartDisease", "num"]

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error("Kolom target tidak ditemukan di dataset")
    st.stop()

st.success(f"Kolom target terdeteksi: {target_col}")

# =========================
# PREPROCESSING
# =========================
X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode data kategorik
X = pd.get_dummies(X, drop_first=True)

# Simpan kolom training
feature_columns = X.columns

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"🎯 Akurasi Model: **{accuracy * 100:.2f}%**")

# =========================
# FORM INPUT USER
# =========================
st.subheader("📝 Masukkan Data Pasien")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Umur", 20, 100, 50)
        sex = st.selectbox("Jenis Kelamin", ["M", "F"])
        cp = st.selectbox("Tipe Nyeri Dada (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 120)
        chol = st.number_input("Kolesterol", 100, 600, 200)
        fbs = st.selectbox("Gula Darah > 120 mg/dl", [0, 1])

    with col2:
        restecg = st.selectbox("Hasil EKG", [0, 1, 2])
        thalach = st.number_input("Detak Jantung Maks", 60, 220, 150)
        exang = st.selectbox("Nyeri Saat Olahraga", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Jumlah Pembuluh (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", ["normal", "fixed", "reversible"])

    submitted = st.form_submit_button("🔍 Prediksi")

# =========================
# PREDIKSI
# =========================
if submitted:
    # Data input user
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Encode input
    input_data = pd.get_dummies(input_data)

    # Samakan kolom dengan data training
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    st.subheader("📌 Hasil Prediksi")

    if prediction == 1:
        st.error("⚠️ Berisiko Mengalami Penyakit Jantung")
    else:
        st.success("✅ Tidak Berisiko Penyakit Jantung")
