import os
import hashlib
from PIL import Image

# 📂 Dossier des images
DATASET_DIR = "data/extracted"

# 🔍 Vérification et suppression des doublons
def clean_images():
    hashes = set()
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            file_path = os.path.join(root, file)

            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"🗑️ Suppression : {file_path} (fichier non image)")
                os.remove(file_path)
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in hashes:
                    print(f"🗑️ Suppression : {file_path} (doublon détecté)")
                    os.remove(file_path)
                else:
                    hashes.add(file_hash)
            except:
                print(f"⚠️ Suppression : {file_path} (fichier corrompu)")
                os.remove(file_path)

# 🚀 Exécution
clean_images()
print("✅ Nettoyage terminé : fichiers corrompus et doublons supprimés.")
