#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import cv2

# 📂 Configuration
TRAIN_CSV = "data/audio/train_image_audio_fusion_mapping.csv"
IMAGE_MODEL_PATH = "models/image_classifier.keras"
AUDIO_MODEL_PATH = "models/audio_classifier.keras"
OUTPUT_MODEL_PATH = "models/image_audio_fusion_model_v2.keras"

# 🔄 Chargement des modèles individuels
print("🔄 Chargement des modèles individuels...")
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
print("✅ Modèles chargés.")

# 🔄 Extraction des caractéristiques
image_feature_extractor = Model(inputs=image_model.input, outputs=image_model.layers[-2].output)
audio_feature_extractor = Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output)

# 📂 Chargement des données
print("🔄 Chargement du mapping d'entraînement...")
train_df = pd.read_csv(TRAIN_CSV)

X_images, X_audio, y_labels = [], [], []

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)

def preprocess_audio(audio_path):
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(1, 64, 64, 1)

for _, row in train_df.iterrows():
    img = preprocess_image(row["image_path"])
    aud = preprocess_audio(row["audio_path"])
    if img is None or aud is None:
        continue
    X_images.append(img)
    X_audio.append(aud)
    y_labels.append(row["label"])

X_images = np.vstack(X_images)
X_audio = np.vstack(X_audio)
y_labels = np.array(y_labels)

# 🔄 Extraction des caractéristiques
X_image_features = image_feature_extractor.predict(X_images)
X_audio_features = audio_feature_extractor.predict(X_audio)

# 🔄 Division train/test
X_train_img, X_test_img, X_train_aud, X_test_aud, y_train, y_test = train_test_split(
    X_image_features, X_audio_features, y_labels, test_size=0.2, random_state=42)

# 🔄 Modèle de fusion entraîné
image_input = Input(shape=(256,))
audio_input = Input(shape=(256,))
merged = concatenate([image_input, audio_input])

x = Dense(128, activation="relu")(merged)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
output = Dense(3, activation="softmax")(x)

fusion_model_v2 = Model(inputs=[image_input, audio_input], outputs=output)
fusion_model_v2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fusion_model_v2.summary()

# 🔄 Entraînement
fusion_model_v2.fit([X_train_img, X_train_aud], y_train, epochs=10, batch_size=16, validation_data=([X_test_img, X_test_aud], y_test))

# 🔄 Sauvegarde du modèle
fusion_model_v2.save(OUTPUT_MODEL_PATH)
print(f"✅ Modèle de fusion sauvegardé : {OUTPUT_MODEL_PATH}")
