#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import random

# 📂 Dossiers d’images et audio pour le test
IMAGE_TEST_FOLDER = "data/images/cleaned/test_set"
AUDIO_TEST_FOLDER = "data/audio/cleaned/test"

# 📄 Fichier de sortie
OUTPUT_CSV = "data/audio/test_image_audio_mapping.csv"

# 🔄 Générer le mapping propre
def associate_images_with_sounds(image_dir, audio_dir, output_csv):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "audio_path"])

        for category in ["cats", "dogs"]:
            img_dir = os.path.join(image_dir, category)
            audio_dir_cat = os.path.join(audio_dir, category)

            if not os.path.exists(img_dir) or not os.path.exists(audio_dir_cat):
                print(f"⚠️ Dossier introuvable : {img_dir} ou {audio_dir_cat}")
                continue

            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
            audio_files = [f for f in os.listdir(audio_dir_cat) if f.lower().endswith(".wav")]

            if not image_files or not audio_files:
                print(f"⚠️ Pas assez de fichiers dans {img_dir} ou {audio_dir_cat}")
                continue

            for img in image_files:
                selected_audio = random.choice(audio_files)
                
                # ✅ Vérification du label de l’audio via son nom
                audio_label = "cat" if "cat_" in selected_audio else "dog"
                # Vérification que l'image et l'audio correspondent à la catégorie
                if category == "cats" and audio_label != "cat":
                    continue
                if category == "dogs" and audio_label != "dog":
                    continue

                writer.writerow([
                    os.path.join(img_dir, img),
                    os.path.join(audio_dir_cat, selected_audio)
                ])

associate_images_with_sounds(IMAGE_TEST_FOLDER, AUDIO_TEST_FOLDER, OUTPUT_CSV)
print("✅ Mapping corrigé et généré avec succès !")
