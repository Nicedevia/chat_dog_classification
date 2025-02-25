#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv

# Paramètre de sous-échantillonnage pour les paires discordantes (ici 20% seulement)
discordant_sample_rate = 0.20

# Répertoires d'images et d'audio d'entraînement
image_train_cats = "data/images/cleaned/training_set/cats"
image_train_dogs = "data/images/cleaned/training_set/dogs"
audio_train_cats = "data/audio/cleaned/train/cats"
audio_train_dogs = "data/audio/cleaned/train/dogs"

mapping_rows = []

# --- Paires concordantes ---
# Chat (label 0)
if os.path.exists(image_train_cats) and os.path.exists(audio_train_cats):
    cat_images = [os.path.join(image_train_cats, f) for f in os.listdir(image_train_cats)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    cat_audios = [os.path.join(audio_train_cats, f) for f in os.listdir(audio_train_cats)
                  if f.lower().endswith(".wav")]
    for img in cat_images:
        if cat_audios:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 0])  # 0 = Chat

# Chien (label 1)
if os.path.exists(image_train_dogs) and os.path.exists(audio_train_dogs):
    dog_images = [os.path.join(image_train_dogs, f) for f in os.listdir(image_train_dogs)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    dog_audios = [os.path.join(audio_train_dogs, f) for f in os.listdir(audio_train_dogs)
                  if f.lower().endswith(".wav")]
    for img in dog_images:
        if dog_audios:
            aud = random.choice(dog_audios)
            mapping_rows.append([img, aud, 1])  # 1 = Chien

# --- Paires discordantes (Erreur, label 2) ---
# Chat image + Chien audio
if os.path.exists(image_train_cats) and os.path.exists(audio_train_dogs):
    for img in cat_images:
        if dog_audios and random.random() < discordant_sample_rate:
            aud = random.choice(dog_audios)
            mapping_rows.append([img, aud, 2])  # 2 = Erreur

# Chien image + Chat audio
if os.path.exists(image_train_dogs) and os.path.exists(audio_train_cats):
    for img in dog_images:
        if cat_audios and random.random() < discordant_sample_rate:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 2])  # 2 = Erreur

# Calcul du nombre de paires par label
count_chat   = sum(1 for row in mapping_rows if row[2] == 0)
count_chien  = sum(1 for row in mapping_rows if row[2] == 1)
count_erreur = sum(1 for row in mapping_rows if row[2] == 2)

# Sauvegarder le mapping dans un fichier CSV
output_csv = "data/audio/train_image_audio_fusion_mapping.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "audio_path", "label"])
    writer.writerows(mapping_rows)

print(f"Mapping généré : {len(mapping_rows)} paires enregistrées dans {output_csv}")
print(f"Répartition des classes : Chat: {count_chat}, Chien: {count_chien}, Erreur: {count_erreur}")
