import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Berita Hoaks",
    layout="centered"
)

st.title("📰 Aplikasi Klasifikasi Berita Hoaks")
st.write("Deteksi berita **Hoaks** atau **Fakta** menggunakan NLP")

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    # Dataset hoaks (Kominfo / Komdigi)
    hoaks = pd.read_csv("komdigi_hoaks.csv")

    # Gabungkan kolom teks yang benar
    hoaks["text"] = (
        hoaks["title"].fillna("") + " " +
        hoaks["body_text"].fillna("")
    )

    hoaks["label"] = 1  # 1 = Hoaks

    # Dataset fakta (contoh, sebaiknya diganti dataset asli)
    fakta = pd.DataFrame({
        "text": [
            "Presiden meresmikan pembangunan jalan tol baru di Jawa Tengah",
            "Pemerintah mengumumkan jadwal libur nasional tahun 2025",
            "Kementerian Kesehatan merilis data terbaru kasus demam berdarah",
            "Bank Indonesia mempertahankan suku bunga acuan"
        ],
        "label": [0, 0, 0, 0]  # 0 = Fakta
    })

    # Gabungkan hoaks + fakta
    data = pd.concat(
        [hoaks[["text", "label"]], fakta],
        ignore_index=True
    )

    return data

data = load_data()

st.write("📊 Jumlah Data:", data.shape[0])
st.write("📌 Distribusi Label:")
st.write(data["label"].value_counts())

# =========================
# PREPROCESSING TEKS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)      # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hapus simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# =========================
# TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=2
)

X = vectorizer.fit_transform(data["text"])
y = data["label"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN MODEL
# =========================
model = MultinomialNB()
model.fit(X_train, y_train)

# =========================
# EVALUASI MODEL
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"🎯 Akurasi Model: **{accuracy * 100:.2f}%**")

# =========================
# INPUT USER
# =========================
st.subheader("🔍 Uji Berita")

user_input = st.text_area(
    "Masukkan teks berita yang ingin diuji:",
    height=200
)

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu.")
    else:
        clean_input = clean_text(user_input)
        vector_input = vectorizer.transform([clean_input])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.error("❌ Berita ini terdeteksi sebagai **HOAKS**")
        else:
            st.success("✅ Berita ini terdeteksi sebagai **FAKTA**")