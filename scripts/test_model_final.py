#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 📂 Définition des chemins
MODEL_PATH = "models/image_audio_fusion_model.keras"
TEST_CSV = "data/audio/test_image_audio_mapping.csv"

# 🚀 Chargement du modèle fusionné
print("🔄 Chargement du modèle fusionné...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modèle chargé avec succès !")

# 🎨 Prétraitement de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)

# 🎵 Prétraitement de l'audio : on charge le spectrogramme pré-généré
def preprocess_audio(audio_path):
    # Transformation du chemin : de "cleaned" vers "spectrograms" et extension .png
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(1, 64, 64, 1)

print("🔄 Prétraitement des données de test...")
test_df = pd.read_csv(TEST_CSV)
X_images, X_audio, y_true = [], [], []

for _, row in test_df.iterrows():
    img_path, audio_path = row["image_path"], row["audio_path"]

    if not os.path.exists(img_path) or not os.path.exists(audio_path):
        continue

    proc_img = preprocess_image(img_path)
    proc_audio = preprocess_audio(audio_path)
    if proc_img is None or proc_audio is None:
        continue

    X_images.append(proc_img)
    X_audio.append(proc_audio)

    # 🔍 Définition du label :
    # Si image et audio indiquent la même catégorie, on attribue le label (0 = Chat, 1 = Chien)
    # Sinon, on attribue le label 2 (Erreur)
    if "cats" in img_path.lower() and "cats" in audio_path.lower():
        y_true.append(0)
    elif "dogs" in img_path.lower() and "dogs" in audio_path.lower():
        y_true.append(1)
    else:
        y_true.append(2)

if len(X_images) == 0 or len(X_audio) == 0:
    print("❌ Aucun échantillon de test valide trouvé.")
    exit()

X_images = np.vstack(X_images)
X_audio = np.vstack(X_audio)
y_true = np.array(y_true)

print("🔄 Prédictions en cours...")
y_pred_probs = model.predict([X_images, X_audio])
y_pred = np.argmax(y_pred_probs, axis=1)

# Détection des labels uniques présents dans le jeu de test
unique_labels = np.unique(y_true)
label_names = {0: "Chat", 1: "Chien", 2: "Erreur"}
target_names = [label_names[l] for l in unique_labels]

print("\n📌 Rapport de classification :")
print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[label_names[l] for l in unique_labels],
            yticklabels=[label_names[l] for l in unique_labels])
plt.xlabel("Prédictions")
plt.ylabel("Vraies Classes")
plt.title("Matrice de Confusion")
plt.show()

accuracy = np.mean(y_pred == y_true) * 100
print(f"\n🎯 Test Accuracy (Fusion): {accuracy:.2f}%")

# 💾 Sauvegarde des résultats
test_results_path = "test_results_v7.csv"
test_df["prediction"] = y_pred
test_df.to_csv(test_results_path, index=False)
print(f"\n✅ Résultats sauvegardés dans {test_results_path}")
