import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung (SVM)",
    layout="centered"
)

st.title("🫀 Prediksi Penyakit Jantung")
st.write("Model terbaik berdasarkan perbandingan: **Support Vector Machine (SVM)**")

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

st.success(f"Kolom target: **{target_col}**")

# =========================
# ENCODING DATA KATEGORIKAL
# =========================
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# =========================
# SPLIT FITUR & LABEL
# =========================
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
svm_model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

# =========================
# EVALUASI MODEL
# =========================
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# =========================
# TAMPILKAN HASIL
# =========================
st.subheader("📈 Evaluasi Model (SVM)")

st.metric("Accuracy", f"{accuracy:.2%}")

st.write("### Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - SVM")
st.pyplot(fig)

# =========================
# FORM PREDIKSI USER
# =========================
st.subheader("🧪 Prediksi Manual")

with st.form("prediction_form"):
    inputs = {}
    for col in X.columns:
        value = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
        inputs[col] = value

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    prediction = svm_model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Berisiko Penyakit Jantung")
    else:
        st.success("✅ Tidak Berisiko Penyakit Jantung")
