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

# 📂 Définition des chemins mis à jour
MODEL_PATH = "models/image_audio_fusion_model.keras"
TEST_IMAGE_FOLDER = "data/images/cleaned/test_set"      # images de test
TEST_AUDIO_FOLDER = "data/audio/cleaned/test"            # audios de test

# 📦 Charger le modèle avec cache pour éviter de le recharger à chaque exécution
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 🎨 Préparation de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)

# 🎵 Préparation de l'audio (conversion en spectrogramme)
def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
    except Exception as e:
        st.error(f"Erreur de chargement audio : {e}")
        return None
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Création explicite d'une figure et d'un axe
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
    return spec_img.reshape(1, 64, 64, 1)

# 🔊 Jouer un son avec Pygame (au clic seulement)
def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

# 🎲 Sélection aléatoire d'un fichier dans un dossier
def get_random_file(folder):
    files = os.listdir(folder)
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# ✅ Initialisation des états de session pour stocker image et son
if "image_path" not in st.session_state:
    st.session_state["image_path"] = None
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

# 🏠 **Interface Streamlit**
st.title("🐱🐶 Classification Chat / Chien")
st.write("Sélectionnez une image et un son du jeu de test, puis lancez la prédiction.")

col1, col2 = st.columns(2)

# 📸 **Sélectionner une image**
with col1:
    st.subheader("Image")
    # Sélection par catégorie
    image_category = st.radio("Catégorie d'image :", ["Chat", "Chien"], key="image_cat")
    # Sélection manuelle
    selected_image = st.file_uploader("📂 Charger une image", type=["jpg", "png"], key="image_upload")
    # Bouton pour sélection aléatoire
    if st.button("🔀 Image Aléatoire", key="rand_image"):
        folder = os.path.join(TEST_IMAGE_FOLDER, "cats" if image_category == "Chat" else "dogs")
        random_img = get_random_file(folder)
        if random_img:
            st.session_state["image_path"] = random_img
    # Mise à jour de l'image si l'utilisateur en charge une
    if selected_image:
        temp_path = f"temp_upload_{selected_image.name}"
        with open(temp_path, "wb") as f:
            f.write(selected_image.getbuffer())
        st.session_state["image_path"] = temp_path
    # Affichage de l'image sélectionnée
    if st.session_state["image_path"]:
        st.image(st.session_state["image_path"], caption="Image sélectionnée", use_column_width=True)

# 🎵 **Sélectionner un son**
with col2:
    st.subheader("Son")
    # Sélection par catégorie
    audio_category = st.radio("Catégorie de son :", ["Chat", "Chien"], key="audio_cat")
    # Sélection manuelle
    selected_audio = st.file_uploader("📂 Charger un fichier audio", type=["wav"], key="audio_upload")
    # Bouton pour sélection aléatoire
    if st.button("🔀 Son Aléatoire", key="rand_audio"):
        folder = os.path.join(TEST_AUDIO_FOLDER, "cats" if audio_category == "Chat" else "dogs")
        random_audio = get_random_file(folder)
        if random_audio:
            st.session_state["audio_path"] = random_audio
    # Mise à jour du son si l'utilisateur en charge un
    if selected_audio:
        temp_audio_path = f"temp_upload_{selected_audio.name}"
        with open(temp_audio_path, "wb") as f:
            f.write(selected_audio.getbuffer())
        st.session_state["audio_path"] = temp_audio_path
    # Affichage du lecteur audio
    if st.session_state["audio_path"]:
        st.audio(st.session_state["audio_path"])
        # Bouton pour jouer le son
        if st.button("▶️ Écouter le Son", key="play_audio"):
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
            prediction = model.predict([X_image, X_audio])
            class_index = np.argmax(prediction)
            # Assurez-vous que l'ordre des classes correspond à votre entraînement :
            # Par exemple : 0 = Chat, 1 = Chien, 2 = Erreur
            class_labels = ["🐱 Chat", "🐶 Chien", "❌ Erreur"]
            confidence = f"{np.max(prediction) * 100:.2f}%"
            st.success(f"✅ **Prédiction : {class_labels[class_index]}** (Confiance : {confidence})")
