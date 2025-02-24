#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile
import shutil

# ====================================================
# Définition des chemins
# ====================================================
ZIP_FILE = "data/cat-and-dog.zip"
EXTRACT_DIR = "data/extracted"

# ====================================================
# Étape 1 : Extraction de l’archive ZIP
# ====================================================
def extract_zip(zip_path, extract_dir):
    if not os.path.exists(zip_path):
        print(f"❌ Erreur : le fichier {zip_path} n'existe pas!")
        return
    print(f"📦 Extraction de {zip_path} vers {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ Extraction terminée.")

# ====================================================
# Étape 2 : Aplatir la structure des dossiers imbriqués
# ====================================================
def flatten_structure():
    # Pour chaque ensemble (training_set et test_set)
    for split in ["training_set", "test_set"]:
        split_path = os.path.join(EXTRACT_DIR, split)
        if not os.path.exists(split_path):
            continue

        # (A) Cas où un dossier dupliqué existe : data/extracted/{split}/{split}
        nested_split = os.path.join(split_path, split)
        if os.path.isdir(nested_split):
            print(f"🔄 Aplatissement du dossier imbriqué {nested_split}...")
            for item in os.listdir(nested_split):
                source = os.path.join(nested_split, item)
                destination = os.path.join(split_path, item)
                shutil.move(source, destination)
            os.rmdir(nested_split)
            print(f"✅ Dossier {nested_split} supprimé.")

        # (B) Pour chaque catégorie, s'il y a un doublon : data/extracted/{split}/cats/cats ou dogs/dogs
        for category in ["cats", "dogs"]:
            cat_path = os.path.join(split_path, category)
            if os.path.isdir(cat_path):
                nested_cat = os.path.join(cat_path, category)
                if os.path.isdir(nested_cat):
                    print(f"🔄 Aplatissement de {nested_cat} dans {cat_path}...")
                    for file in os.listdir(nested_cat):
                        source = os.path.join(nested_cat, file)
                        destination = os.path.join(cat_path, file)
                        shutil.move(source, destination)
                    shutil.rmtree(nested_cat)
                    print(f"✅ Dossier {nested_cat} supprimé.")

# ====================================================
# Étape 3 : Réorganiser les images dans les bons dossiers
# ====================================================
def reorganize_images():
    """
    Déplace toutes les images trouvées dans n'importe quel sous-dossier de
    chaque ensemble (training_set et test_set) vers les dossiers cibles :
    data/extracted/{split}/cats ou .../dogs
    """
    for split in ["training_set", "test_set"]:
        split_path = os.path.join(EXTRACT_DIR, split)
        if not os.path.exists(split_path):
            print(f"❌ Le dossier {split_path} est introuvable.")
            continue

        # Créer les dossiers cibles si besoin
        for category in ["cats", "dogs"]:
            target_dir = os.path.join(split_path, category)
            os.makedirs(target_dir, exist_ok=True)

        # Parcourir récursivement le dossier split
        for root, dirs, files in os.walk(split_path):
            # Ignorer les dossiers cibles déjà organisés
            if os.path.basename(root) in ["cats", "dogs"]:
                continue
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    lower_file = file.lower()
                    if "cat" in lower_file:
                        dest_folder = os.path.join(split_path, "cats")
                    elif "dog" in lower_file:
                        dest_folder = os.path.join(split_path, "dogs")
                    else:
                        continue
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_folder, file)
                    # Si le fichier n'est pas déjà à la bonne place
                    if os.path.abspath(src_file) != os.path.abspath(dest_file):
                        if os.path.exists(dest_file):
                            print(f"⚠️ Doublon pour {file} dans {dest_folder}, fichier ignoré.")
                        else:
                            shutil.move(src_file, dest_file)
                            print(f"📸 Déplacé {file} vers {dest_folder}.")

# ====================================================
# Étape 4 : Supprimer les dossiers vides et fichiers inutiles
# ====================================================
def remove_empty_dirs():
    """
    Supprime tous les dossiers vides dans EXTRACT_DIR ainsi que les fichiers
    indésirables comme _DS_Store.
    """
    for dirpath, dirnames, filenames in os.walk(EXTRACT_DIR, topdown=False):
        # Supprimer les fichiers inutiles
        for f in filenames:
            if f == "_DS_Store" or f.startswith('.'):
                os.remove(os.path.join(dirpath, f))
        # Supprimer le dossier s'il est vide
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            print(f"🗑️ Dossier vide supprimé : {dirpath}")

# ====================================================
# Étape 5 : Vérification de la structure finale
# ====================================================
def verify_structure():
    print("\n📊 Structure finale dans data/extracted :")
    for split in ["training_set", "test_set"]:
        for category in ["cats", "dogs"]:
            path = os.path.join(EXTRACT_DIR, split, category)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png"))]
                print(f"{path}: {len(files)} images")
            else:
                print(f"❌ {path} n'existe pas.")

# ====================================================
# Fonction principale
# ====================================================
def main():
    extract_zip(ZIP_FILE, EXTRACT_DIR)
    flatten_structure()
    reorganize_images()
    remove_empty_dirs()
    verify_structure()
    print("\n✅ Traitement terminé. La structure dans 'data/extracted' contient uniquement les dossiers nécessaires.")

if __name__ == "__main__":
    main()
