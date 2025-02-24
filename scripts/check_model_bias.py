import os
import tensorflow as tf
import numpy as np
import cv2
import librosa

# 📂 Définition des chemins
TEST_IMAGE_DIR = "data/extracted/test_set"
TEST_AUDIO_DIR = "data/audio/augmented"
MODEL_PATH = "models/audio_image_classifier_v2.keras"

# ✅ Charger le modèle
model = tf.keras.models.load_model(MODEL_PATH)

# 📌 **Prétraitement des images et sons**
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=2)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    spectrogram_db = cv2.resize(spectrogram_db, (64, 64)) / 255.0
    return spectrogram_db.reshape(1, 64, 64, 1)

# 📊 Tester plusieurs images et sons
def test_bias():
    total = 0
    correct = 0
    errors = 0

    categories = ["cats", "dogs"]
    for category in categories:
        img_dir = os.path.join(TEST_IMAGE_DIR, category)
        audio_dir = os.path.join(TEST_AUDIO_DIR, category)

        images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        audios = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

        for img_file, audio_file in zip(images[:10], audios[:10]):  # Tester 10 images + sons par catégorie
            img_path = os.path.join(img_dir, img_file)
            audio_path = os.path.join(audio_dir, audio_file)

            img_input = preprocess_image(img_path)
            audio_input = preprocess_audio(audio_path)

            # 🔍 Faire la prédiction
            prediction = model.predict([img_input, audio_input])[0][0]
            confidence = round(float(prediction) * 100, 2)

            # Détermination de la classe
            predicted_label = "🐶 Chien" if confidence > 50 else "🐱 Chat"
            expected_label = "🐱 Chat" if "cats" in img_path else "🐶 Chien"

            # Vérifier si la prédiction est correcte
            if predicted_label == expected_label:
                correct += 1
            else:
                errors += 1
                print(f"🚨 Erreur → Prédit {predicted_label} ({confidence}%) alors que c'était {expected_label}.")

            total += 1

    print("\n📊 **Résumé du test :**")
    print(f"✅ Corrects : {correct}/{total} ({(correct/total) * 100:.2f}%)")
    print(f"❌ Erreurs : {errors}/{total} ({(errors/total) * 100:.2f}%)")

if __name__ == "__main__":
    test_bias()
