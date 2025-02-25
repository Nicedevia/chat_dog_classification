import os
import random
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import pygame

# 📂 Définition des chemins
FUSION_MODEL_PATH = "models/image_audio_fusion_model_v2.keras"
IMAGE_MODEL_PATH = "models/image_classifier.keras"
AUDIO_MODEL_PATH = "models/audio_classifier.keras"

TEST_IMAGE_FOLDER = "data/images/cleaned/test_set"
TEST_AUDIO_FOLDER = "data/audio/cleaned/test"

# 📦 Charger les modèles
@st.cache_resource
def load_models():
    fusion_model = tf.keras.models.load_model(FUSION_MODEL_PATH)
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

    # Extraire les feature extractors
    image_feature_extractor = tf.keras.Model(inputs=image_model.input, outputs=image_model.layers[-2].output)
    audio_feature_extractor = tf.keras.Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output)

    return fusion_model, image_feature_extractor, audio_feature_extractor

fusion_model, image_feature_extractor, audio_feature_extractor = load_models()

# 🎨 Préparation de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)

    # 🔥 Extraire les features de l'image
    image_features = image_feature_extractor.predict(img)
    return image_features

# 🎵 Préparation de l'audio (conversion en spectrogramme)
def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
    except Exception as e:
        st.error(f"Erreur de chargement audio : {e}")
        return None

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.axis("off")

    temp_img_path = "temp_spectrogram.png"
    fig.savefig(temp_img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    spec_img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    spec_img = spec_img.reshape(1, 64, 64, 1)

    # 🔥 Extraire les features de l'audio
    audio_features = audio_feature_extractor.predict(spec_img)
    return audio_features

# 🔊 Jouer un son avec Pygame
def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

# 🎲 Sélection aléatoire d'un fichier dans un dossier
def get_random_file(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# ✅ Initialisation des états de session
if "image_path" not in st.session_state:
    st.session_state["image_path"] = None
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

# 🏠 **Interface Streamlit**
st.title("🐱🐶 Classification Chat / Chien")
st.write("Sélectionnez une image et un son pour tester la classification.")

col1, col2 = st.columns(2)

# 📸 **Sélectionner une image**
with col1:
    st.subheader("Image")
    image_category = st.radio("Catégorie d'image :", ["Chat", "Chien"], key="image_cat", index=0)

    # Ne change l'image que si elle n'a pas encore été sélectionnée
    if st.session_state["image_path"] is None:
        folder = os.path.join(TEST_IMAGE_FOLDER, "cats" if image_category == "Chat" else "dogs")
        st.session_state["image_path"] = get_random_file(folder)

    # Bouton pour recharger une nouvelle image
    if st.button("🔄 Changer l'image"):
        folder = os.path.join(TEST_IMAGE_FOLDER, "cats" if image_category == "Chat" else "dogs")
        st.session_state["image_path"] = get_random_file(folder)

    # Affichage de l'image
    if st.session_state["image_path"]:
        st.image(st.session_state["image_path"], caption="Image sélectionnée", use_column_width=True)

# 🎵 **Sélectionner un son**
with col2:
    st.subheader("Son")
    audio_category = st.radio("Catégorie de son :", ["Chat", "Chien"], key="audio_cat", index=0)

    # Ne change le son que si aucun n'a encore été sélectionné
    if st.session_state["audio_path"] is None:
        folder = os.path.join(TEST_AUDIO_FOLDER, "cats" if audio_category == "Chat" else "dogs")
        st.session_state["audio_path"] = get_random_file(folder)

    # Bouton pour recharger un nouveau son
    if st.button("🔄 Changer le son"):
        folder = os.path.join(TEST_AUDIO_FOLDER, "cats" if audio_category == "Chat" else "dogs")
        st.session_state["audio_path"] = get_random_file(folder)

    # Affichage du lecteur audio
    if st.session_state["audio_path"]:
        st.audio(st.session_state["audio_path"])
        if st.button("▶️ Écouter le Son"):
            play_audio(st.session_state["audio_path"])

# 🔍 **Prédiction**
if st.button("🔮 Prédire"):
    if not st.session_state["image_path"] or not st.session_state["audio_path"]:
        st.warning("⚠️ Sélectionnez une image et un son avant de prédire.")
    else:
        X_image = preprocess_image(st.session_state["image_path"])
        X_audio = preprocess_audio(st.session_state["audio_path"])
        if X_image is None or X_audio is None:
            st.error("Erreur lors du prétraitement. Vérifiez vos fichiers.")
        else:
            # 🧠 Fusion des features et prédiction
            prediction = fusion_model.predict([X_image, X_audio])
            class_index = np.argmax(prediction)
            class_labels = ["🐱 Chat", "🐶 Chien", "❌ Erreur"]
            confidence = f"{np.max(prediction) * 100:.2f}%"
            st.success(f"✅ **Prédiction : {class_labels[class_index]}** (Confiance : {confidence})")
