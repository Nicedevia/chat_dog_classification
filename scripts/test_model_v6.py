import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 📂 Définition des chemins
MODEL_PATH = "models/audio_image_classifier_v6.keras"
TEST_CSV = "data/audio/test_image_audio_mapping.csv"

# Vérifier l'existence des fichiers
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Modèle introuvable : {MODEL_PATH}")
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"🚨 Fichier de test introuvable : {TEST_CSV}")

# ✅ Charger le modèle
print("🔄 Chargement du modèle V6...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modèle chargé avec succès !")

# 🎛 Paramètres d'image et audio
IMG_SIZE = (64, 64)
SR = 22050  # Fréquence d'échantillonnage
N_MELS = 128  # Nombre de bandes mel
DURATION = 2  # Durée en secondes

# 🎨 Fonction pour convertir un fichier audio en spectrogramme
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Convertir en format image
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_axis_off()
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
    
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Convertir en échelle de gris et redimensionner
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE) / 255.0
    return img_resized

# 📂 Chargement des données de test
test_df = pd.read_csv(TEST_CSV)
X_audio, X_images, y_true = [], [], []

print("🔄 Prétraitement des données de test...")



import matplotlib.pyplot as plt

def visualize_input(image_path, spectrogram_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Affichage de l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image")

    # Affichage du spectrogramme
    spec = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
    axes[1].imshow(spec, cmap="gray")
    axes[1].set_title("Spectrogramme")

    plt.show()

# Avant d’envoyer au modèle, affichons les données chargées
visualize_input(image_path, spectrogram_path)

for _, row in test_df.iterrows():
    img_path, audio_path = row["image_path"], row["audio_path"]

    # Vérifier l'existence des fichiers
    if not os.path.exists(img_path) or not os.path.exists(audio_path):
        print(f"⚠️ Fichier manquant : {img_path} ou {audio_path}")
        continue

    # Chargement et prétraitement de l'image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE) / 255.0

    # Conversion de l'audio en spectrogramme
    spec_img = audio_to_spectrogram(audio_path)

    X_images.append(img)
    X_audio.append(spec_img)

    # Définir la classe cible (0 = Chat, 1 = Chien, 2 = Erreur)
    if "cats" in img_path and "cats" in audio_path:
        y_true.append(0)  # Chat
    elif "dogs" in img_path and "dogs" in audio_path:
        y_true.append(1)  # Chien
    else:
        y_true.append(2)  # Erreur (mismatch image-son)

# 📦 Conversion en tenseurs
X_audio = np.array(X_audio).reshape(-1, 64, 64, 1)
X_images = np.array(X_images).reshape(-1, 64, 64, 1)
y_true = np.array(y_true)

# 🚀 Prédiction avec le modèle
print("🔄 Prédictions en cours...")
y_pred = model.predict([X_images, X_audio])
y_pred = np.argmax(y_pred, axis=1)  # Convertir en labels 0, 1 ou 2

# 📊 Calcul des métriques
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=["Chat", "Chien", "Erreur"])

# 📜 Affichage des résultats
print("\n📊 **Résumé des Performances :**")
print(f"🎯 **Test Accuracy:** {accuracy:.2%}\n")

print("📌 **Matrice de Confusion :**")
print(conf_matrix)

print("\n📜 **Rapport de Classification :**")
print(class_report)

# 💾 Sauvegarde des résultats
results_df = pd.DataFrame({"Image_Path": test_df["image_path"], "Audio_Path": test_df["audio_path"], "True_Label": y_true, "Predicted_Label": y_pred})
results_df.to_csv("test_results_v6.csv", index=False)
print("✅ Résultats sauvegardés dans test_results_v6.csv")