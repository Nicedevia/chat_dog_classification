#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile
import shutil

# ====================================================
# Définition des chemins
# ====================================================
# Pour l'audio, on utilise l'ancien dataset
audio_zip = "data/audio-cats-and-dogs.zip"
# Pour les images, on utilise le dataset amélioré (25 000 images)
image_zip = "data/Cat_Dog_data.zip"

audio_extract_folder = "data/audio"
image_extract_folder = "data/extracted"

# ====================================================
# Fonction d'extraction sécurisée
# ====================================================
def extract_zip(zip_path, extract_to):
    """Extrait un ZIP dans le dossier spécifié si le fichier existe."""
    if not os.path.exists(zip_path):
        print(f"❌ Erreur : {zip_path} n'existe pas !")
        return
    print(f"📦 Extraction de {zip_path} dans {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraction terminée pour {zip_path}.")

# ====================================================
# Extraction des archives
# ====================================================
print("🔄 Extraction des archives...")
extract_zip(audio_zip, audio_extract_folder)
extract_zip(image_zip, image_extract_folder)
print("✅ Extraction des archives terminée.")

# ====================================================
# Réorganisation des images depuis le dataset amélioré
# ====================================================
def reorganize_extracted_images():
    """
    Reorganise les images extraites du fichier Cat_Dog_data.zip.
    On attend que l'extraction crée un dossier :
        data/extracted/Cat_Dog_data/
            train/
                cat/   -->  images d'entraînement pour "cat"
                dog/   -->  images d'entraînement pour "dog"
            test/
                cat/   -->  images de test pour "cat"
                dog/   -->  images de test pour "dog"
    Les images seront déplacées vers :
        data/extracted/training_set/cats
        data/extracted/training_set/dogs
        data/extracted/test_set/cats
        data/extracted/test_set/dogs
    Et le dossier source sera supprimé.
    """
    src_root = os.path.join(image_extract_folder, "Cat_Dog_data")
    if not os.path.exists(src_root):
        print(f"❌ Dossier source pour les images introuvable : {src_root}")
        return

    # Définition des dossiers de destination
    dest_train_cats = os.path.join(image_extract_folder, "training_set", "cats")
    dest_train_dogs = os.path.join(image_extract_folder, "training_set", "dogs")
    dest_test_cats  = os.path.join(image_extract_folder, "test_set", "cats")
    dest_test_dogs  = os.path.join(image_extract_folder, "test_set", "dogs")

    for folder in [dest_train_cats, dest_train_dogs, dest_test_cats, dest_test_dogs]:
        os.makedirs(folder, exist_ok=True)

    # Déplacer les images d'entraînement
    src_train_cat = os.path.join(src_root, "train", "cat")
    src_train_dog = os.path.join(src_root, "train", "dog")
    if os.path.exists(src_train_cat):
        for file in os.listdir(src_train_cat):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.move(os.path.join(src_train_cat, file), dest_train_cats)
        print(f"📸 Images de train/cat déplacées vers {dest_train_cats}")
    else:
        print(f"❌ Dossier introuvable : {src_train_cat}")

    if os.path.exists(src_train_dog):
        for file in os.listdir(src_train_dog):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.move(os.path.join(src_train_dog, file), dest_train_dogs)
        print(f"📸 Images de train/dog déplacées vers {dest_train_dogs}")
    else:
        print(f"❌ Dossier introuvable : {src_train_dog}")

    # Déplacer les images de test
    src_test_cat = os.path.join(src_root, "test", "cat")
    src_test_dog = os.path.join(src_root, "test", "dog")
    if os.path.exists(src_test_cat):
        for file in os.listdir(src_test_cat):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.move(os.path.join(src_test_cat, file), dest_test_cats)
        print(f"📸 Images de test/cat déplacées vers {dest_test_cats}")
    else:
        print(f"❌ Dossier introuvable : {src_test_cat}")

    if os.path.exists(src_test_dog):
        for file in os.listdir(src_test_dog):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.move(os.path.join(src_test_dog, file), dest_test_dogs)
        print(f"📸 Images de test/dog déplacées vers {dest_test_dogs}")
    else:
        print(f"❌ Dossier introuvable : {src_test_dog}")

    # Supprimer le dossier source une fois le transfert terminé
    shutil.rmtree(src_root)
    print(f"🗑️ Dossier source supprimé : {src_root}")

# ====================================================
# Organisation des fichiers audio (inchangée)
# ====================================================
def organize_audio_files():
    """
    Déplace les fichiers audio vers les dossiers appropriés :
        data/audio/cleaned/train/cats, train/dogs, test/cats, test/dogs
    """
    base_folder = os.path.join(audio_extract_folder, "cats_dogs")
    if not os.path.exists(base_folder):
        print("❌ Dossier `cats_dogs` non trouvé. Vérifiez l'extraction.")
        return

    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")

    correct_train_cats = os.path.join(audio_extract_folder, "cleaned", "train", "cats")
    correct_train_dogs = os.path.join(audio_extract_folder, "cleaned", "train", "dogs")
    correct_test_cats  = os.path.join(audio_extract_folder, "cleaned", "test", "cats")
    correct_test_dogs  = os.path.join(audio_extract_folder, "cleaned", "test", "dogs")

    for folder in [correct_train_cats, correct_train_dogs, correct_test_cats, correct_test_dogs]:
        os.makedirs(folder, exist_ok=True)

    for folder, subset in [(train_folder, "train"), (test_folder, "test")]:
        if not os.path.exists(folder):
            continue
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith(".wav"):
                        if "cat" in file.lower() and subset == "train":
                            dest_folder = correct_train_cats
                        elif "dog" in file.lower() and subset == "train":
                            dest_folder = correct_train_dogs
                        elif "cat" in file.lower() and subset == "test":
                            dest_folder = correct_test_cats
                        elif "dog" in file.lower() and subset == "test":
                            dest_folder = correct_test_dogs
                        else:
                            continue
                        dest_path = os.path.join(dest_folder, file)
                        if os.path.exists(dest_path):
                            print(f"⚠️ Doublon détecté, fichier ignoré : {dest_path}")
                        else:
                            shutil.move(os.path.join(subfolder_path, file), dest_folder)
                shutil.rmtree(subfolder_path)
    print("✅ Tous les fichiers audio ont été déplacés correctement.")

# ====================================================
# Suppression des dossiers inutiles et vérification finale
# ====================================================
def remove_empty_dirs_and_verify():
    # Suppression des dossiers vides dans data/extracted
    for dirpath, dirnames, filenames in os.walk(image_extract_folder, topdown=False):
        for d in dirnames:
            dpath = os.path.join(dirpath, d)
            if not os.listdir(dpath):
                os.rmdir(dpath)
                print(f"🗑️ Dossier vide supprimé : {dpath}")
    # Suppression des dossiers inutiles dans data/audio (exemple: cats_dogs)
    cats_dogs_folder = os.path.join(audio_extract_folder, "cats_dogs")
    if os.path.exists(cats_dogs_folder):
        shutil.rmtree(cats_dogs_folder)
        print(f"🗑️ Dossier inutile supprimé : {cats_dogs_folder}")

    # Vérification finale
    paths = [
        os.path.join(audio_extract_folder, "cleaned", "train", "cats"),
        os.path.join(audio_extract_folder, "cleaned", "train", "dogs"),
        os.path.join(audio_extract_folder, "cleaned", "test", "cats"),
        os.path.join(audio_extract_folder, "cleaned", "test", "dogs"),
        os.path.join(image_extract_folder, "training_set", "cats"),
        os.path.join(image_extract_folder, "training_set", "dogs"),
        os.path.join(image_extract_folder, "test_set", "cats"),
        os.path.join(image_extract_folder, "test_set", "dogs"),
    ]
    print("\n📊 Structure finale :")
    for path in paths:
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"{path}: {count} fichiers")
        else:
            print(f"❌ Dossier manquant : {path}")

# ====================================================
# Fonction principale
# ====================================================
def main():
    print("🔄 Extraction du dataset...")
    # Extraction audio et image
    extract_zip(audio_zip, audio_extract_folder)
    extract_zip(image_zip, image_extract_folder)
    print("✅ Extraction terminée.")

    print("🔄 Réorganisation des images...")
    reorganize_extracted_images()

    print("🔄 Organisation des fichiers audio...")
    organize_audio_files()

    print("🔄 Suppression des dossiers vides et vérification finale...")
    remove_empty_dirs_and_verify()

    print("\n✅ Extraction et corrections terminées avec succès !")

if __name__ == "__main__":
    main()
