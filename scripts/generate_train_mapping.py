#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv

# Répertoires d'images et d'audio pour l'entraînement
image_train_cats = "data/images/cleaned/training_set/cats"
image_train_dogs = "data/images/cleaned/training_set/dogs"
audio_train_cats = "data/audio/cleaned/train/cats"
audio_train_dogs = "data/audio/cleaned/train/dogs"

mapping_rows = []

# Générer le mapping pour les chats
if os.path.exists(image_train_cats) and os.path.exists(audio_train_cats):
    image_files = [os.path.join(image_train_cats, f) for f in os.listdir(image_train_cats) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    audio_files = [os.path.join(audio_train_cats, f) for f in os.listdir(audio_train_cats) if f.lower().endswith(".wav")]
    for img in image_files:
        if audio_files:
            aud = random.choice(audio_files)
            mapping_rows.append([img, aud])

# Générer le mapping pour les chiens
if os.path.exists(image_train_dogs) and os.path.exists(audio_train_dogs):
    image_files = [os.path.join(image_train_dogs, f) for f in os.listdir(image_train_dogs) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    audio_files = [os.path.join(audio_train_dogs, f) for f in os.listdir(audio_train_dogs) if f.lower().endswith(".wav")]
    for img in image_files:
        if audio_files:
            aud = random.choice(audio_files)
            mapping_rows.append([img, aud])

# Sauvegarder le mapping dans un fichier CSV
mapping_csv = "data/audio/train_image_audio_mapping.csv"
os.makedirs(os.path.dirname(mapping_csv), exist_ok=True)
with open(mapping_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "audio_path"])
    writer.writerows(mapping_rows)

print(f"Mapping généré avec succès : {len(mapping_rows)} paires. Fichier sauvegardé dans {mapping_csv}")
