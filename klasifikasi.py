import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =========================
# JUDUL APLIKASI
# =========================
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")
st.title("📰 Aplikasi Klasifikasi Berita Hoaks")
st.write("Deteksi berita Hoaks atau Fakta menggunakan NLP")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    hoaks = pd.read_csv("komdigi_hoaks.csv")
    st.write(hoaks.columns)

    # Gabungkan kolom teks (sesuaikan dengan dataset kamu)
    hoaks["text"] = hoaks["title"].fillna("") + " " + hoaks["content"].fillna("")
    hoaks["label"] = 1  # Hoaks

    # CONTOH DATA FAKTA (WAJIB ADA)
    # GANTI dengan dataset fakta asli
    fakta = pd.DataFrame({
        "text": [
            "Presiden meresmikan pembangunan jalan tol baru",
            "Pemerintah mengumumkan jadwal libur nasional"
        ],
        "label": [0, 0]
    })

    data = pd.concat([hoaks[["text", "label"]], fakta], ignore_index=True)
    return data


data = load_data()

st.write("📊 Jumlah Data:", data.shape[0])
st.write(data["label"].value_counts())

# =========================
# PREPROCESSING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = MultinomialNB()
model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("🎯 Akurasi Model:", round(accuracy * 100, 2), "%")

# =========================
# INPUT USER
# =========================
st.subheader("🔍 Uji Berita")
user_input = st.text_area("Masukkan teks berita:")

if st.button("Prediksi"):
    clean_input = clean_text(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.error("❌ Berita ini terdeteksi sebagai HOAKS")
    else:
        st.success("✅ Berita ini terdeteksi sebagai FAKTA")
