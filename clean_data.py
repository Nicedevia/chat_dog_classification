#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

# Liste des dossiers à nettoyer (relatifs à la racine "data")
FOLDERS_TO_CLEAN = [
    "data/audio/cleaned",
    "data/audio/spectrograms",
    "data/extracted",
    "data/images/cleaned"
]

def clean_folder(folder_path):
    """Supprime tous les fichiers (et dossiers vides) dans le dossier donné."""
    if not os.path.exists(folder_path):
        print(f"⚠️ Dossier non trouvé : {folder_path}")
        return

    # Parcourt le dossier de manière récursive
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"❌ Supprimé : {file_path}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la suppression de {file_path} : {e}")
    # Optionnel : Supprimer les dossiers vides
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                # Si le dossier est vide, le supprimer
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"🗑️ Dossier vide supprimé : {dir_path}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la suppression du dossier {dir_path} : {e}")

def clean_data():
    print("🧹 Nettoyage complet du projet (à partir de 'data')...")
    for folder in FOLDERS_TO_CLEAN:
        print(f"🔍 Nettoyage du dossier : {folder}")
        clean_folder(folder)
    print("✅ Nettoyage terminé !")

if __name__ == "__main__":
    clean_data()
