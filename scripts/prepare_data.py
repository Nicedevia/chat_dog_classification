import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Définition des paramètres
IMG_SIZE = (128, 128)  # Taille des images
BATCH_SIZE = 32  # Nombre d'images chargées en mémoire à la fois
DATASET_DIR = "data/extracted"

# Préparation du DataLoader avec augmentation des images
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisation des pixels (0-1)
    validation_split=0.2,  # 20% des images pour validation
    horizontal_flip=True,  # Retourner les images horizontalement (augmentation)
    zoom_range=0.2  # Zoom aléatoire
)

# Chargement des données d'entraînement
train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "training_set"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Classification binaire (chien vs chat)
    subset='training'
)

# Chargement des données de validation
val_generator = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "training_set"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("✅ Données prêtes pour l'entraînement !")
