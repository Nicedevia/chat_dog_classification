#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 📂 Configuration
TEST_CSV = "data/audio/test_image_audio_mapping.csv"
IMAGE_MODEL_PATH = "models/image_classifier.keras"
AUDIO_MODEL_PATH = "models/audio_classifier.keras"
FUSION_MODEL_PATH = "models/image_audio_fusion_model_v2.keras"

# 🔄 Chargement des modèles
print("🔄 Chargement des modèles individuels et du modèle fusionné...")
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
fusion_model = tf.keras.models.load_model(FUSION_MODEL_PATH)
print("✅ Modèles chargés.")

# 🔄 Extraction des caractéristiques des modèles individuels
image_feature_extractor = tf.keras.Model(inputs=image_model.input, outputs=image_model.layers[-2].output)
audio_feature_extractor = tf.keras.Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output)

# 📂 Chargement des données de test
print("🔄 Chargement du mapping de test...")
test_df = pd.read_csv(TEST_CSV)

X_images, X_audio, y_true = [], [], []

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

for _, row in test_df.iterrows():
    img = preprocess_image(row["image_path"])
    aud = preprocess_audio(row["audio_path"])
    if img is None or aud is None:
        continue
    X_images.append(img)
    X_audio.append(aud)
    y_true.append(row["label"])

X_images = np.vstack(X_images)
X_audio = np.vstack(X_audio)
y_true = np.array(y_true)

print(f"🔄 Nombre d'échantillons de test utilisés : {X_images.shape[0]}")

# 🔄 Extraction des caractéristiques
X_image_features = image_feature_extractor.predict(X_images)
X_audio_features = audio_feature_extractor.predict(X_audio)

# 🔄 Prédictions avec le modèle fusionné
y_pred_probs = fusion_model.predict([X_image_features, X_audio_features])
y_pred_final = np.argmax(y_pred_probs, axis=1)

# 🔄 Évaluation du modèle
label_names = {0: "Chat", 1: "Chien", 2: "Erreur"}
target_names = [label_names[0], label_names[1], label_names[2]]

print("\n📌 Rapport de classification :")
print(classification_report(y_true, y_pred_final, target_names=target_names))

conf_matrix = confusion_matrix(y_true, y_pred_final)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Prédictions")
plt.ylabel("Vraies Classes")
plt.title("Matrice de Confusion")
plt.show()

accuracy = np.mean(y_pred_final == y_true) * 100
print(f"\n🎯 Test Accuracy (Fusion Model V2): {accuracy:.2f}%")

# 🔄 Sauvegarde des résultats
OUTPUT_RESULTS = "test_results_fusion_v2.csv"
test_df["prediction"] = y_pred_final
test_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"\n✅ Résultats sauvegardés dans {OUTPUT_RESULTS}")
