import os
import zipfile
import kaggle  # Assurez-vous que le package est installé : pip install kaggle
import os
os.environ['KAGGLE_CONFIG_DIR'] = "config"

# 📂 Définition des chemins et variables
KAGGLE_CONFIG_DIR = os.path.expanduser("~/.kaggle")  # Dossier de configuration Kaggle
KAGGLE_JSON_PATH = "config/kaggle.json"  # Fichier JSON contenant la clé API
DATASET_NAME = "tongpython/cat-and-dog"  # Nom du dataset Kaggle
DOWNLOAD_DIR = "data"  # Dossier de téléchargement
EXTRACT_DIR = "data/extracted"  # Dossier de sortie des fichiers extraits

def setup_kaggle_api():
    """ 🔐 Configure l'API Kaggle en copiant kaggle.json dans le bon répertoire """
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)  # Création du dossier ~/.kaggle si inexistant
    kaggle_json_target = os.path.join(KAGGLE_CONFIG_DIR, "kaggle.json")
    
    if not os.path.exists(kaggle_json_target):
        print("🛠️ Copie de kaggle.json dans ~/.kaggle/")
        os.system(f"cp {KAGGLE_JSON_PATH} {kaggle_json_target}")
        os.chmod(kaggle_json_target, 600)  # Sécurisation du fichier
    else:
        print("✅ Clé API Kaggle déjà configurée.")

def download_kaggle_data():
    """ ⬇️ Télécharge et extrait les données Kaggle """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)  # Création du dossier data
    
    # Vérifie si les données existent déjà pour éviter de les re-télécharger
    dataset_zip_path = os.path.join(DOWNLOAD_DIR, "cat-and-dog.zip")
    
    if not os.path.exists(dataset_zip_path):
        print(f"⬇️ Téléchargement du dataset {DATASET_NAME}...")
        kaggle.api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_DIR, unzip=False)
        print("✅ Téléchargement terminé !")
    else:
        print("📂 Le fichier existe déjà, pas de nouveau téléchargement.")

    extract_files(dataset_zip_path)

def extract_files(zip_path):
    """ 📦 Décompresse les fichiers téléchargés """
    os.makedirs(EXTRACT_DIR, exist_ok=True)  # Création du dossier d'extraction

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
        print(f"✅ Fichiers extraits dans {EXTRACT_DIR}")

if __name__ == "__main__":
    setup_kaggle_api()
    download_kaggle_data()
