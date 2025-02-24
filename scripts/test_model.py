import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2
import sys
import os

# Charger le modèle
MODEL_PATH = "models/cat_dog_classifier.h5"  # Vérifie bien le chemin
model = keras.models.load_model(MODEL_PATH)

# Classes
class_names = ["Cat", "Dog"]

# Charger une image pour test
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Erreur : L'image {image_path} n'existe pas.")
        return
    
    # Charger et prétraiter l'image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Redimensionne à la taille du modèle
    img = img / 255.0  # Normalisation
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension batch

    # Prédiction
    prediction = model.predict(img)
    predicted_class = class_names[int(prediction[0] > 0.5)]  # Seuil à 0.5

    print(f"✅ Prédiction : {predicted_class}")

import pygame

def play_sound(label):
    """Joue un son en fonction de la prédiction"""
    sound_path = "sounds/cat.wav" if label == "Cat" else "sounds/dog.wav"
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()

# Appeler la fonction après la prédiction
play_sound(predicted_label)


# Vérifier si un fichier image a été fourni
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_model.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
