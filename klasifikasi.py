import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="centered"
)

st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.write(
    "Aplikasi ini menggunakan **Support Vector Machine (SVM)** "
    "yang dipilih karena memiliki performa terbaik berdasarkan hasil pengujian."
)

# =========================
# CATATAN PENTING
# =========================
st.info(
    "📌 **Catatan Penting**\n\n"
    "Aplikasi ini menggunakan model Support Vector Machine (SVM) yang dipilih "
    "berdasarkan hasil perbandingan dengan Random Forest. "
    "Model SVM memiliki kemampuan deteksi risiko yang lebih baik "
    "berdasarkan nilai recall dan F1-score.\n\n"
    "Hasil prediksi bersifat edukatif dan tidak menggantikan pemeriksaan dokter."
    "- Beberapa data diolah dalam bentuk **angka (0 atau 1)** karena model "
    "Machine Learning hanya dapat memproses data numerik.\n"
    "- **Pengguna cukup memilih opsi yang tersedia**, sistem akan mengubahnya "
    "secara otomatis.\n\n"
    "**Jenis Kelamin:**\n"
    "- 0 = Perempuan\n"
    "- 1 = Laki-laki"
)



# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# =========================
# TENTUKAN TARGET
# =========================
possible_targets = ["target", "output", "HeartDisease", "num"]
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

# =========================
# ENCODING DATASET
# =========================
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALING (WAJIB UNTUK SVM)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# TRAIN MODEL SVM
# =========================
model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
model.fit(X_train_scaled, y_train)

# =========================
# FORM INPUT USER (RAMAH)
# =========================
st.subheader("🧪 Masukkan Data Anda")

age = st.number_input("Usia (tahun)", 1, 120, 30)

sex = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
sex_value = 0 if sex == "Perempuan" else 1

chest_pain = st.selectbox(
    "Jenis Nyeri Dada",
    ["Tidak ada / tidak khas", "Ringan", "Sedang", "Berat"]
)
chest_pain_value = [
    "Tidak ada / tidak khas", "Ringan", "Sedang", "Berat"
].index(chest_pain)

resting_bp = st.number_input("Tekanan Darah Saat Istirahat (mmHg)", 80, 250, 120)
cholesterol = st.number_input("Kadar Kolesterol (mg/dL)", 100, 600, 200)

fasting_bs = st.selectbox("Gula Darah Puasa", ["Normal", "Tinggi"])
fasting_bs_value = 0 if fasting_bs == "Normal" else 1

resting_ecg = st.selectbox(
    "Hasil EKG Istirahat", ["Normal", "Kelainan Ringan", "Kelainan Berat"]
)
resting_ecg_value = ["Normal", "Kelainan Ringan", "Kelainan Berat"].index(resting_ecg)

max_hr = st.number_input("Denyut Jantung Maksimum", 60, 220, 150)

exercise_angina = st.selectbox(
    "Nyeri Dada Saat Aktivitas", ["Tidak", "Ya"]
)
exercise_angina_value = 0 if exercise_angina == "Tidak" else 1

oldpeak = st.number_input("Penurunan Segmen ST", -5.0, 10.0, 0.0)

st_slope = st.selectbox(
    "Kemiringan Segmen ST", ["Naik", "Datar", "Turun"]
)
st_slope_value = ["Naik", "Datar", "Turun"].index(st_slope)

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi Risiko"):
    input_data = pd.DataFrame([[
        age, sex_value, chest_pain_value, resting_bp,
        cholesterol, fasting_bs_value, resting_ecg_value,
        max_hr, exercise_angina_value, oldpeak, st_slope_value
    ]], columns=X.columns)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

if prediction == 1:
    st.warning(
        "⚠️ **Hasil Prediksi SVM**\n\n"
        "Berdasarkan pengolahan data menggunakan model Support Vector Machine (SVM), "
        "data yang dimasukkan memiliki **pola yang menyerupai pasien dengan risiko penyakit jantung**.\n\n"
        "⚠️ Hasil ini bersifat **edukatif** dan **tidak menggantikan diagnosis medis**."
    )
else:
    st.success(
        "✅ **Hasil Prediksi SVM**\n\n"
        "Berdasarkan model SVM, data yang dimasukkan **tidak menunjukkan pola risiko penyakit jantung**."
    )

# =========================
# ===== BAGIAN PENGUJIAN =====
# =========================
st.markdown("---")
st.subheader("📊 Hasil Pengujian Model")

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Akurasi Model (SVM)", f"{accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title("Confusion Matrix - SVM")
st.pyplot(fig)

st.write("### Dataset (Cuplikan)")
st.dataframe(df.head())
