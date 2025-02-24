import os
import zipfile
import shutil

# 📂 Définition des chemins
audio_zip = "data/audio-cats-and-dogs.zip"
image_zip = "data/cat-and-dog.zip"
audio_extract_folder = "data/audio"
image_extract_folder = "data/extracted"

# 🔄 Fonction d'extraction sécurisée
def extract_zip(zip_path, extract_to):
    """ Extrait un ZIP dans le dossier spécifié si le fichier existe """
    if not os.path.exists(zip_path):
        print(f"❌ Erreur : {zip_path} n'existe pas !")
        return

    print(f"📦 Extraction de {zip_path} dans {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraction terminée pour {zip_path}.")

# ✅ Étape 1 : Ré-extraire les fichiers ZIP pour éviter la suppression complète
extract_zip(audio_zip, audio_extract_folder)
extract_zip(image_zip, image_extract_folder)

# 🔄 Correction des sous-dossiers mal extraits (ex: `test/test`)
def fix_directory_structure(base_dir, expected_dirs):
    """ Corrige les structures de répertoires mal extraites """
    for expected_dir in expected_dirs:
        nested_dir = os.path.join(base_dir, expected_dir, expected_dir)
        correct_dir = os.path.join(base_dir, expected_dir)

        if os.path.exists(nested_dir):
            print(f"🔄 Correction : {nested_dir} → {correct_dir}")
            for filename in os.listdir(nested_dir):
                src_path = os.path.join(nested_dir, filename)
                dest_path = os.path.join(correct_dir, filename)

                if os.path.exists(dest_path):
                    print(f"⚠️ Fichier déjà existant, ignoré : {dest_path}")
                else:
                    shutil.move(src_path, correct_dir)

            shutil.rmtree(nested_dir)
            print(f"✅ Dossier supprimé : {nested_dir}")

# ✅ Étape 2 : Corrige `test_set` et `training_set`
fix_directory_structure(image_extract_folder, ["test_set", "training_set"])

# 🔄 Organisation et correction des fichiers audio
def organize_audio_files():
    """ Déplace les fichiers audio vers `train/cats`, `train/dogs`, `test/cats`, `test/dogs` et supprime `cats_dogs/`. """
    base_folder = os.path.join(audio_extract_folder, "cats_dogs")

    if not os.path.exists(base_folder):
        print("❌ Dossier `cats_dogs` non trouvé. Vérifiez l'extraction.")
        return

    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")

    correct_train_cats = os.path.join(audio_extract_folder, "train", "cats")
    correct_train_dogs = os.path.join(audio_extract_folder, "train", "dogs")
    correct_test_cats = os.path.join(audio_extract_folder, "test", "cats")
    correct_test_dogs = os.path.join(audio_extract_folder, "test", "dogs")

    os.makedirs(correct_train_cats, exist_ok=True)
    os.makedirs(correct_train_dogs, exist_ok=True)
    os.makedirs(correct_test_cats, exist_ok=True)
    os.makedirs(correct_test_dogs, exist_ok=True)

    for folder, category in [(train_folder, "train"), (test_folder, "test")]:
        if not os.path.exists(folder):
            continue

        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".wav"):
                        dest_folder = correct_train_cats if "cat" in file and category == "train" else \
                                      correct_train_dogs if "dog" in file and category == "train" else \
                                      correct_test_cats if "cat" in file and category == "test" else \
                                      correct_test_dogs
                        
                        dest_path = os.path.join(dest_folder, file)

                        if os.path.exists(dest_path):
                            print(f"⚠️ Doublon détecté, fichier ignoré : {dest_path}")
                        else:
                            shutil.move(os.path.join(subfolder_path, file), dest_path)

                shutil.rmtree(subfolder_path)

    print("✅ Tous les fichiers audio ont été déplacés correctement.")

# ✅ Étape 3 : Exécuter la correction de l'organisation audio
organize_audio_files()

# 🔄 Supprimer `cats_dogs/` après déplacement
cats_dogs_folder = os.path.join(audio_extract_folder, "cats_dogs")
if os.path.exists(cats_dogs_folder):
    print(f"🗑 Suppression du dossier inutile : {cats_dogs_folder}")
    shutil.rmtree(cats_dogs_folder)

# 🔄 Vérification finale et affichage des fichiers
def verify_data():
    """ Vérifie si les fichiers sont bien placés """
    paths = [
        os.path.join(audio_extract_folder, "train", "cats"),
        os.path.join(audio_extract_folder, "train", "dogs"),
        os.path.join(audio_extract_folder, "test", "cats"),
        os.path.join(audio_extract_folder, "test", "dogs"),
        os.path.join(image_extract_folder, "training_set", "cats"),
        os.path.join(image_extract_folder, "training_set", "dogs"),
        os.path.join(image_extract_folder, "test_set", "cats"),
        os.path.join(image_extract_folder, "test_set", "dogs"),
    ]

    print("\n📊 **Vérification des fichiers après extraction :**")
    for path in paths:
        if os.path.exists(path):
            num_files = len(os.listdir(path))
            print(f"📂 {path} → {num_files} fichiers trouvés")
        else:
            print(f"❌ Dossier manquant : {path}")

# ✅ Étape 4 : Vérification des fichiers et affichage des résultats
verify_data()

print("✅ Extraction et corrections terminées avec succès !")
