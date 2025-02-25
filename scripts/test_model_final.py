#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix  # roc_curve, auc supprimés
from sklearn.preprocessing import label_binarize

# ----- Configuration -----
# Chemin du mapping de test (assurez-vous que ce fichier contient les colonnes "image_path" et "audio_path")
TEST_CSV = "data/audio/test_image_audio_mapping.csv"

# Chemins des modèles individuels
IMAGE_MODEL_PATH = "models/image_classifier.keras"
AUDIO_MODEL_PATH = "models/audio_classifier.keras"

# ----- Chargement des modèles individuels -----
print("🔄 Chargement du modèle image...")
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
print("✅ Modèle image chargé.")

print("🔄 Chargement du modèle audio...")
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
print("✅ Modèle audio chargé.")

# ----- Fonctions de prétraitement -----
def preprocess_image(image_path):
    """Charge et prétraite une image en niveaux de gris."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)

def preprocess_audio(audio_path):
    """
    Charge le spectrogramme pré-généré correspondant au fichier audio.
    On suppose que le chemin audio est de la forme :
        data/audio/cleaned/...
    et que le spectrogramme est pré-généré dans :
        data/audio/spectrograms/...
    avec l'extension .wav remplacée par .png.
    """
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(1, 64, 64, 1)

# ----- Chargement du mapping de test -----
print("🔄 Chargement du mapping de test...")
test_df = pd.read_csv(TEST_CSV)

X_images, X_audio = [], []
y_true = []  # Étiquette "virtuelle" dérivée du nom de fichier

for _, row in test_df.iterrows():
    img_path = row["image_path"]
    audio_path = row["audio_path"]
    if not os.path.exists(img_path) or not os.path.exists(audio_path):
        continue
    X_img = preprocess_image(img_path)
    X_aud = preprocess_audio(audio_path)
    if X_img is None or X_aud is None:
        continue
    X_images.append(X_img)
    X_audio.append(X_aud)
    # Définir le label à partir des chemins :
    # Si les deux chemins contiennent "cats", label = 0 ; s'ils contiennent "dogs", label = 1 ; sinon, label = 2.
    if "cats" in img_path.lower() and "cats" in audio_path.lower():
        y_true.append(0)
    elif "dogs" in img_path.lower() and "dogs" in audio_path.lower():
        y_true.append(1)
    else:
        y_true.append(2)

if len(X_images) == 0 or len(X_audio) == 0:
    print("❌ Aucun échantillon de test valide trouvé.")
    exit()

# Convertir les listes en tenseurs
X_images = np.vstack(X_images)
X_audio = np.vstack(X_audio)
y_true = np.array(y_true)

print(f"🔄 Nombre d'échantillons de test utilisés : {X_images.shape[0]}")

# ----- Prédictions individuelles -----
# Prédiction du modèle image (sortie sigmoïde, seuil 0.5)
image_preds_prob = image_model.predict(X_images)
image_preds = (image_preds_prob > 0.5).astype(int).reshape(-1)

# Prédiction du modèle audio (sortie sigmoïde, seuil 0.5)
audio_preds_prob = audio_model.predict(X_audio)
audio_preds = (audio_preds_prob > 0.5).astype(int).reshape(-1)

# ----- Fusion des prédictions -----
# Si les deux prédictions (image et audio) sont identiques, la prédiction finale est cette valeur (0 ou 1).
# Sinon, on considère que le modèle est en désaccord et on définit la prédiction finale comme 2 (Erreur).
y_pred_final = []
for img_pred, aud_pred in zip(image_preds, audio_preds):
    if img_pred == aud_pred:
        y_pred_final.append(img_pred)
    else:
        y_pred_final.append(2)
y_pred_final = np.array(y_pred_final)

# ----- Évaluation -----
label_names = {0: "Chat", 1: "Chien", 2: "Erreur"}
target_names = [label_names[0], label_names[1], label_names[2]]

print("\n📌 Rapport de classification :")
print(classification_report(y_true, y_pred_final, target_names=target_names))

conf_matrix = confusion_matrix(y_true, y_pred_final, labels=[0, 1, 2])
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Prédictions")
plt.ylabel("Vraies Classes")
plt.title("Matrice de Confusion")
plt.show()


accuracy = np.mean(y_pred_final == y_true) * 100
print(f"\n🎯 Test Accuracy (Ensemble): {accuracy:.2f}%")

# ----- Sauvegarde des résultats -----
OUTPUT_RESULTS = "test_results_v7.csv"
test_df["prediction"] = y_pred_final
test_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"\n✅ Résultats sauvegardés dans {OUTPUT_RESULTS}")
