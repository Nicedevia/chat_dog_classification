#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import os
from sklearn.metrics import classification_report, accuracy_score

# 📂 Chargement du modèle Audio
model = tf.keras.models.load_model("models/audio_classifier.keras")

# 📂 Chargement du mapping de test
test_csv = "data/audio/test_image_audio_mapping.csv"
test_df = pd.read_csv(test_csv)

X_audio_test, y_true = [], []

for _, row in test_df.iterrows():
    audio_path = row["audio_path"]
    # On transforme le chemin audio en chemin vers le spectrogramme correspondant
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        continue

    # 🎵 Chargement et prétraitement du spectrogramme audio
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        continue
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    X_audio_test.append(spec_img)

    # 🏷️ Définition du label réel (0 = Chat, 1 = Chien)
    if "cats" in audio_path.lower():
        y_true.append(0)
    else:
        y_true.append(1)

# 📦 Conversion en tenseurs
X_audio_test = np.array(X_audio_test).reshape(-1, 64, 64, 1)
y_true = np.array(y_true)

# 🚀 Prédiction (attention : modèle binaire avec une sortie sigmoïde)
y_pred_probs = model.predict(X_audio_test)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# 📊 Rapport de classification et accuracy
print("📌 Rapport de classification (Audio Only) :")
print(classification_report(y_true, y_pred, target_names=["Chat", "Chien"]))
accuracy = accuracy_score(y_true, y_pred)
print(f"🎯 Test Accuracy (Audio Only): {accuracy:.2%}")
