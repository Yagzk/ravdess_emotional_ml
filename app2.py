import streamlit as st
import joblib
import librosa
import numpy as np

# ================================
# Fonksiyon: Özellik çıkarımı
# ================================
def aug_extract_features_single(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rms, centroid])

# ================================
# Pickle dosyalarını yükle
# ================================
@st.cache_resource
def load_model_parts():
    scaler = joblib.load("aug_scaler.pkl")
    pca = joblib.load("aug_pca.pkl")
    model = joblib.load("aug_best_model.pkl")
    le = joblib.load("aug_labelencoder.pkl")
    return scaler, pca, model, le

AUG_scaler, AUG_pca, AUG_model, AUG_le = load_model_parts()

# ================================
# Streamlit arayüzü
# ================================
st.title("🎤 Duygu Tanıma: Ses Dosyanı Yükle ve Test Et")

uploaded_file = st.file_uploader("Bir ses dosyası yükleyin (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Dosyayı oku ve özellik çıkar
    y, sr = librosa.load(uploaded_file, sr=48000)
    features = aug_extract_features_single(y, sr).reshape(1, -1)

    # Ölçekle + PCA + tahmin
    features_scaled = AUG_scaler.transform(features)
    features_pca = AUG_pca.transform(features_scaled)
    pred = AUG_model.predict(features_pca)
    pred_label = AUG_le.inverse_transform(pred)[0]

    st.success(f"🔊 **Tahmin edilen duygu:** {pred_label}")
