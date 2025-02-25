#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import random

# Paramètre : taux de paires discordantes (ici 20 %)
discordant_sample_rate = 0.20

# Dossiers de test d'après l'arborescence fournie
IMAGE_TEST_FOLDER = "data/images/cleaned/test_set"
AUDIO_TEST_FOLDER = "data/audio/cleaned/test"

# Fichier de sortie pour le mapping de test
OUTPUT_CSV = "data/audio/test_image_audio_mapping.csv"

def generate_test_mapping(image_dir, audio_dir, output_csv):
    mapping_rows = []
    
    # Vérification des répertoires pour le débogage
    for cat in ["cats", "dogs"]:
        img_dir = os.path.join(image_dir, cat)
        aud_dir = os.path.join(audio_dir, cat)
        print(f"{cat.upper()} - Images: {len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0}, Audios: {len(os.listdir(aud_dir)) if os.path.exists(aud_dir) else 0}")
    
    # --- Paires concordantes ---
    # Pour chaque catégorie ("cats" et "dogs")
    for category in ["cats", "dogs"]:
        img_dir = os.path.join(image_dir, category)
        audio_dir_cat = os.path.join(audio_dir, category)
        
        if not os.path.exists(img_dir) or not os.path.exists(audio_dir_cat):
            print(f"⚠️ Répertoire non trouvé : {img_dir} ou {audio_dir_cat}")
            continue
        
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        audio_files = [f for f in os.listdir(audio_dir_cat) if f.lower().endswith(".wav")]
        
        for img in image_files:
            if audio_files:
                aud = random.choice(audio_files)
                # Label 0 pour "cats", 1 pour "dogs"
                label = 0 if category == "cats" else 1
                mapping_rows.append([os.path.join(img_dir, img), os.path.join(audio_dir_cat, aud), label])
    
    # --- Paires discordantes (Erreur, label 2) ---
    # Pour les images de "cats", associer un audio de "dogs"
    cat_img_dir = os.path.join(image_dir, "cats")
    dog_audio_dir = os.path.join(audio_dir, "dogs")
    if os.path.exists(cat_img_dir) and os.path.exists(dog_audio_dir):
        cat_images = [os.path.join(cat_img_dir, f) for f in os.listdir(cat_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        dog_audios = [os.path.join(dog_audio_dir, f) for f in os.listdir(dog_audio_dir) if f.lower().endswith(".wav")]
        for img in cat_images:
            if dog_audios and random.random() < discordant_sample_rate:
                aud = random.choice(dog_audios)
                mapping_rows.append([img, aud, 2])
    
    # Pour les images de "dogs", associer un audio de "cats"
    dog_img_dir = os.path.join(image_dir, "dogs")
    cat_audio_dir = os.path.join(audio_dir, "cats")
    if os.path.exists(dog_img_dir) and os.path.exists(cat_audio_dir):
        dog_images = [os.path.join(dog_img_dir, f) for f in os.listdir(dog_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        cat_audios = [os.path.join(cat_audio_dir, f) for f in os.listdir(cat_audio_dir) if f.lower().endswith(".wav")]
        for img in dog_images:
            if cat_audios and random.random() < discordant_sample_rate:
                aud = random.choice(cat_audios)
                mapping_rows.append([img, aud, 2])
    
    # Sauvegarder le mapping dans un fichier CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "audio_path", "label"])
        writer.writerows(mapping_rows)
    
    # Calcul du nombre de paires par label
    count_chat   = sum(1 for row in mapping_rows if row[2] == 0)
    count_chien  = sum(1 for row in mapping_rows if row[2] == 1)
    count_erreur = sum(1 for row in mapping_rows if row[2] == 2)
    
    print(f"✅ Mapping de test généré : {len(mapping_rows)} paires enregistrées dans {output_csv}")
    print(f"Répartition des classes : Chat: {count_chat}, Chien: {count_chien}, Erreur: {count_erreur}")

if __name__ == "__main__":
    generate_test_mapping(IMAGE_TEST_FOLDER, AUDIO_TEST_FOLDER, OUTPUT_CSV)
