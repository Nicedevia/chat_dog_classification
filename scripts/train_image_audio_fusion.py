#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration et chemins ---
MAPPING_CSV = "data/audio/train_image_audio_fusion_mapping.csv"

# --- Fonctions de prétraitement ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    """
    Charge le spectrogramme pré-généré correspondant au fichier audio.
    On suppose que le chemin de l'audio dans le mapping est de la forme :
      data/audio/cleaned/.../xxx.wav
    et que le spectrogramme est pré-généré dans :
      data/audio/spectrograms/.../xxx.png
    """
    # Transformation du chemin : remplacer "cleaned" par "spectrograms" et ".wav" par ".png"
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(64, 64, 1)

# --- Chargement du mapping ---
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

# --- Chargement des modèles individuels pré-entraînés ---
print("Chargement des modèles individuels pré-entraînés...")
image_model = tf.keras.models.load_model("models/image_classifier.keras")
audio_model = tf.keras.models.load_model("models/audio_classifier.keras")
print("Modèles individuels chargés.")

# Extraction des features (on utilise la sortie de la couche avant la dernière, supposée Dense(256))
image_feature_model = Model(inputs=image_model.input, outputs=image_model.layers[-2].output, name="image_feature_extractor")
audio_feature_model = Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output, name="audio_feature_extractor")

# Optionnel : geler les extracteurs pour se concentrer sur l'entraînement des couches de fusion
image_feature_model.trainable = False
audio_feature_model.trainable = False

# --- Création du modèle fusionné ---
image_input = Input(shape=(64, 64, 1), name="image_input")
audio_input = Input(shape=(64, 64, 1), name="audio_input")

image_features = image_feature_model(image_input)
audio_features = audio_feature_model(audio_input)

combined_features = concatenate([image_features, audio_features], name="fusion_layer")

fc = Dense(128, activation="relu")(combined_features)
fc = Dropout(0.3)(fc)
fc = Dense(64, activation="relu")(fc)
final_output = Dense(3, activation="softmax", name="output_layer")(fc)

fusion_model = Model(inputs=[image_input, audio_input], outputs=final_output, name="fusion_model")
fusion_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fusion_model.summary()

# --- Callbacks personnalisés ---
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"----- Epoch {epoch+1} started -----")
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")
        print(f"----- Epoch {epoch+1} ended: loss={loss:.4f}, accuracy={acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f} -----")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    LoggingCallback(),
    TqdmCallback(verbose=1)
]

# --- Entraînement ---
print("Entraînement du modèle fusionné...")
history = fusion_model.fit([X_images, X_audio], y_labels,
                           epochs=10, validation_split=0.2, batch_size=16,
                           callbacks=callbacks)

# Sauvegarde du modèle fusionné
os.makedirs("models", exist_ok=True)
fusion_model.save("models/image_audio_fusion_model_v2.keras")
print("Modèle fusionné sauvegardé avec succès !")
