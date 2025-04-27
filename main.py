import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
from matplotlib import pyplot as plt
import os
# Genre Labels
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model with cache
@st.cache_resource()
def load_model():
    try:
        model = tf.keras.models.load_model("Trained_model.keras")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# Record audio from mic and save
def record_audio(duration=10, fs=22050, filename="mic_input.wav"):
    st.info("🎙️ Recording from microphone...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.squeeze()
    sf.write(filename, audio, fs)
    st.success("✅ Recording complete.")
    return filename

# Preprocess audio to Mel spectrogram
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        audio_data, _ = librosa.effects.trim(audio_data)

        chunk_duration = 4  # seconds
        overlap_duration = 2
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate

        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            if len(chunk) < chunk_samples:
                continue
            mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel = tf.image.resize(np.expand_dims(mel, axis=-1), target_shape)
            data.append(mel)
        return np.array(data)
    except Exception as e:
        st.error(f"❌ Audio preprocessing failed: {e}")
        return np.array([])

# Predict genre from audio
def model_prediction(X_test):
    model = load_model()
    if model is None or len(X_test) == 0:
        return None
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_elements = unique_elements[counts == np.max(counts)]
    return max_elements[0]

# Sidebar navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Genre Classification"])

# Home page
if app_mode == "Home":
    st.markdown("""
        <style>
        .stApp { background-color: #181646; color: white; }
        h2, h3 { color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🎶 Welcome to the Music Genre Classification System! 🎧")
    st.image("banner.jpg", use_container_width=True)
    st.markdown("""
        Upload or record an audio clip to predict its music genre using deep learning.

        ### Features:
        - 🎧 Real-time microphone input
        - 🔍 Audio file upload
        - ⚙️ Accurate CNN-based classification
    """)

# About page
elif app_mode == "About Project":
    st.markdown("""
Music Genre Recognition using AI/ML
                
📌 Project Description:
The Music Genre Recognition project aims to classify audio files into predefined musical genres using machine learning and deep learning techniques. This system listens to a piece of music and intelligently identifies the genre, just like a human listener would—based on rhythm, timbre, pitch, and other musical features.

By leveraging audio signal processing and AI/ML models, the project extracts key features from sound files, processes them, and accurately predicts genres such as rock, pop, classical, hip-hop, jazz, and more. It serves as a proof-of-concept for real-world applications like music recommendation systems, playlist automation, or music organization tools.

🎯 Objectives:
Automatically classify songs into musical genres.

Explore and compare classical ML algorithms and deep learning approaches.

Understand the relationship between audio features and genre.

Build an interactive frontend or command-line interface for predictions.

🔍 Key Features:
Audio Preprocessing using librosa (Mel Spectrograms, MFCCs, etc.)

Feature Extraction from audio signals

Model Training using ML/DL models (e.g., CNNs, SVMs)

Genre Prediction on unseen music files

Optional Web Interface for uploading and predicting genre

📁 Dataset:
GTZAN Genre Collection

1000 audio tracks (30s each)

10 genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

Source: Marsyas Music Dataset

⚙️ Tools & Technologies:
Python

librosa, numpy, pandas, matplotlib, seaborn

Machine Learning: scikit-learn, xgboost

Deep Learning: TensorFlow or PyTorch (CNNs for spectrograms)

Visualization: matplotlib, seaborn

(Optional) Web App: Flask, Streamlit, or Gradio

🧠 Model Architecture (Deep Learning):
If using CNNs:

Convert audio → Spectrogram images

CNN layers (like image classification)

Dense layers → Softmax output for genre classes

✅ Evaluation Metrics:
Accuracy

Precision, Recall, F1-score

Confusion Matrix

Cross-validation to avoid overfitting

💡 Applications:
Music recommendation systems

Auto-tagging in streaming platforms

Music library organization

DJ tools and smart playlists

Educational tools for music theory

🚀 Future Improvements:
Use larger and more diverse datasets (FMA, Million Song Dataset)

Real-time genre prediction with live audio input

Multi-label classification for fusion genres

Integration with Spotify/Youtube APIs

If you'd like, I can help generate a GitHub README.md, presentation slides, or even code for specific parts like data loading, spectrogram creation, or CNN models. Just let me know!



    """)

# Prediction page
elif app_mode == "Genre Classification":
    st.header("🎤 Genre Prediction")

    input_mode = st.radio("Choose Input Method", ["Upload MP3", "Record via Microphone"])
    audio_path = None

    if input_mode == "Upload MP3":
        uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3", "wav"])
        if uploaded_file is not None:
            os.makedirs("Test_Music", exist_ok=True)
            audio_path = f"Test_Music/{uploaded_file.name}"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.audio(audio_path)

    elif input_mode == "Record via Microphone":
        if st.button("Start Recording"):
            audio_path = record_audio()
            st.audio(audio_path)

    if st.button("Predict Genre") and audio_path:
        with st.spinner("Processing audio..."):
            X_test = load_and_preprocess_data(audio_path)
            if len(X_test) == 0:
                st.error("Could not extract audio features. Try another file.")
            else:
                result_index = model_prediction(X_test)
                if result_index is not None:
                    predicted_genre = GENRES[result_index]
                    st.balloons()
                    st.success(f"🎶 Predicted Genre: {predicted_genre.upper()}")
                else:
                    st.error("Prediction failed. Please check your model and audio input.")
