import os
import random
import csv

# 📂 Définition des chemins
image_folder = "data/extracted/training_set"
test_image_folder = "data/extracted/test_set"
audio_folder = "data/audio/cleaned"

# 📜 Fichiers de sortie
train_output_csv = "data/audio/train_image_audio_mapping.csv"
test_output_csv = "data/audio/test_image_audio_mapping.csv"

# 🔀 Fonction d’association image-son
def associate_images_with_sounds(image_dir, audio_dir, output_csv, dataset_type="train"):
    """ Associe chaque image avec un son et enregistre dans un CSV """
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "audio_path"])

        for category in ["cats", "dogs"]:
            img_dir = os.path.join(image_dir, category)
            audio_category_dir = os.path.join(audio_dir, category)

            # 📂 Vérification de l'existence des dossiers
            if not os.path.exists(img_dir) or not os.path.exists(audio_category_dir):
                print(f"⚠️ Dossier manquant : {img_dir} ou {audio_category_dir}")
                continue

            # 📜 Listes des fichiers
            image_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            audio_files = [f for f in os.listdir(audio_category_dir) if f.endswith(".wav")]

            if not image_files or not audio_files:
                print(f"⚠️ Pas assez de fichiers dans {category} pour l’association ({dataset_type}).")
                continue

            # 🏷️ Associer un fichier son à chaque image
            random.shuffle(audio_files)  # Mélanger pour éviter les biais
            for img in image_files:
                selected_audio = random.choice(audio_files)
                writer.writerow([
                    os.path.join(img_dir, img),
                    os.path.join(audio_category_dir, selected_audio)
                ])

# 🔄 Génération des associations
print("🔄 Génération des associations...")

# 🏋️‍♂️ Association pour l'entraînement
associate_images_with_sounds(image_folder, audio_folder, train_output_csv, dataset_type="train")

# 🎯 Association pour le test
associate_images_with_sounds(test_image_folder, audio_folder, test_output_csv, dataset_type="test")

print(f"✅ Associations terminées !")
print(f"📂 Entraînement : {train_output_csv}")
print(f"📂 Test : {test_output_csv}")
