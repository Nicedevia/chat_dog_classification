#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tqdm.keras import TqdmCallback
import logging

# Configurer le logging pour afficher plus d'informations
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Chemin du mapping d'entraînement
MAPPING_CSV = "data/audio/train_image_audio_fusion_mapping.csv"

# Fonctions de prétraitement
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
    except Exception as e:
        logging.error(f"Erreur de chargement audio pour {audio_path}: {e}")
        return None
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    # Création d'une figure et d'un axe pour éviter les erreurs
    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.axis("off")
    temp_img_path = "temp_spec.png"
    fig.savefig(temp_img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    spec_img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(64, 64, 1)

# Callback personnalisé pour afficher les logs à chaque époque
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"----- Epoch {epoch+1} started -----")
    def on_epoch_end(self, epoch, logs=None):
        print(f"----- Epoch {epoch+1} ended: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}, "
              f"val_loss={logs.get('val_loss'):.4f}, val_accuracy={logs.get('val_accuracy'):.4f} -----")

# Chargement du mapping
df = pd.read_csv(MAPPING_CSV)
print(f"Nombre d'exemples dans le mapping : {len(df)}")

X_images, X_audio, y_labels = [], [], []
for _, row in df.iterrows():
    img = preprocess_image(row["image_path"])
    aud = preprocess_audio(row["audio_path"])
    if img is None or aud is None:
        continue
    X_images.append(img)
    X_audio.append(aud)
    y_labels.append(row["label"])

X_images = np.array(X_images)
X_audio = np.array(X_audio)
y_labels = np.array(y_labels)

print(f"Dataset final : {X_images.shape[0]} exemples")

# Chargement des modèles individuels pré-entraînés
print("Chargement des modèles individuels pré-entraînés...")
image_model = tf.keras.models.load_model("models/image_classifier.keras")
audio_model = tf.keras.models.load_model("models/audio_classifier.keras")
print("Modèles individuels chargés.")

# Extraction des features : on utilise la sortie avant la couche finale (ici supposée être Dense(256))
image_feature_model = Model(inputs=image_model.input, outputs=image_model.layers[-2].output, name="image_feature_extractor")
audio_feature_model = Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output, name="audio_feature_extractor")
# Optionnel : geler ces extracteurs pour concentrer l'entraînement sur les couches de fusion
image_feature_model.trainable = False
audio_feature_model.trainable = False

# Définition des entrées pour chaque modalité
image_input = Input(shape=(64, 64, 1), name="image_input")
audio_input = Input(shape=(64, 64, 1), name="audio_input")

# Extraire les features
image_features = image_feature_model(image_input)
audio_features = audio_feature_model(audio_input)

# Fusionner les features
combined_features = concatenate([image_features, audio_features], name="fusion_layer")

# Couches supplémentaires de fusion
fc = Dense(128, activation="relu")(combined_features)
fc = Dropout(0.3)(fc)
fc = Dense(64, activation="relu")(fc)
# Couche de sortie à 3 neurones pour les 3 classes (Chat, Chien, Erreur)
final_output = Dense(3, activation="softmax", name="output_layer")(fc)

# Création du modèle fusionné
fusion_model = Model(inputs=[image_input, audio_input], outputs=final_output, name="fusion_model")
fusion_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fusion_model.summary()

# Entraînement du modèle fusionné avec callbacks pour plus d'informations
print("Entraînement du modèle fusionné...")
history = fusion_model.fit(
    [X_images, X_audio],
    y_labels,
    epochs=10,
    validation_split=0.2,
    batch_size=16,
    callbacks=[LoggingCallback(), TqdmCallback(verbose=1)]
)

# Sauvegarder le modèle fusionné
os.makedirs("models", exist_ok=True)
fusion_model.save("models/image_audio_fusion_model.keras")
print("Modèle fusionné sauvegardé avec succès !")
