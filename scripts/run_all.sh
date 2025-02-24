#!/bin/bash

set -e  # ⛔ Arrête immédiatement le script en cas d'erreur

echo "🚀 Démarrage du pipeline de traitement et d'entraînement..."

# 📂 Vérification de l'arborescence des fichiers
echo "🔍 Vérification de l’arborescence des fichiers..."
python scripts/check_arborescence.py

# 🧹 Nettoyage des fichiers (images et audio)
echo "🧼 Nettoyage des images et suppression des doublons..."
python scripts/clean_images.py

echo "🧼 Nettoyage des fichiers audio..."
python scripts/clean_audio.py

# 🎵 Génération des spectrogrammes
echo "🎨 Génération des spectrogrammes audio..."
python scripts/generate_spectrograms.py

# 🔄 Génération du mapping image-son
echo "📄 Génération du mapping des fichiers image-audio..."
python scripts/generate_mapping_fixed.py

# 🏋️ Entraînement des modèles
echo "🎓 Entraînement du modèle Image Only..."
python scripts/train_image_only.py

echo "🎓 Entraînement du modèle Audio Only..."
python scripts/train_audio_only.py

echo "🎓 Fusion des modèles Image + Audio..."
python scripts/train_image_audio_fusion.py

# 🛠 Vérification de l'intégrité des données avant test
echo "✅ Vérification de l'intégrité des données..."
python scripts/check_data_integrity.py

# 🔬 Tests des modèles
echo "🧪 Test du modèle basé uniquement sur l’image..."
python scripts/test_model_image_only.py

echo "🧪 Test du modèle basé uniquement sur l’audio..."
python scripts/test_model_audio_only.py

echo "🧪 Test du modèle fusionné Image + Audio..."
python scripts/test_model_final.py

echo "🎉 Tous les tests sont terminés avec succès !"
