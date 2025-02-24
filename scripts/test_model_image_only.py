#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import os
from sklearn.metrics import classification_report, accuracy_score

# 📂 Chargement du modèle Image
model = tf.keras.models.load_model("models/image_classifier.keras")

# 📂 Chargement du mapping de test
test_csv = "data/audio/test_image_audio_mapping.csv"
test_df = pd.read_csv(test_csv)

X_images_test, y_true = [], []

for _, row in test_df.iterrows():
    img_path = row["image_path"]
    if not os.path.exists(img_path):
        continue

    # 📸 Chargement et prétraitement de l'image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64)) / 255.0
    X_images_test.append(img)

    # 🏷️ Définition du label réel (0 = Chat, 1 = Chien)
    if "cats" in img_path.lower():
        y_true.append(0)
    else:
        y_true.append(1)

# 📦 Conversion en tenseurs
X_images_test = np.array(X_images_test).reshape(-1, 64, 64, 1)
y_true = np.array(y_true)

# 🚀 Prédiction (modèle binaire avec sigmoïde)
y_pred_probs = model.predict(X_images_test)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# 📊 Rapport de classification et accuracy
print("📌 Rapport de classification (Image Only) :")
print(classification_report(y_true, y_pred, target_names=["Chat", "Chien"]))
accuracy = accuracy_score(y_true, y_pred)
print(f"🎯 Test Accuracy (Image Only): {accuracy:.2%}")
