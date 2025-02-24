#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
from PIL import Image

# 📂 Dossier des images
DATASET_DIR = "data/extracted"

def clean_images():
    hashes = set()
    total_files = 0
    removed_files = 0

    for root, _, files in os.walk(DATASET_DIR):
        print(f"🔍 Parcours du dossier : {root}")
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            # Vérification de l'extension
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"🗑️ Suppression : {file_path} (fichier non image)")
                os.remove(file_path)
                removed_files += 1
                continue

            try:
                # Vérifier que l'image est valide
                with Image.open(file_path) as img:
                    img.verify()
                # Calculer le hash
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in hashes:
                    print(f"🗑️ Suppression : {file_path} (doublon détecté)")
                    os.remove(file_path)
                    removed_files += 1
                else:
                    hashes.add(file_hash)
            except Exception as e:
                print(f"⚠️ Suppression : {file_path} (fichier corrompu, {e})")
                os.remove(file_path)
                removed_files += 1

    print(f"✅ Nettoyage terminé : {removed_files} fichiers supprimés sur {total_files} fichiers parcourus.")

if __name__ == "__main__":
    clean_images()
