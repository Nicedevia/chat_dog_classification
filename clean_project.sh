#!/bin/bash

echo "🚀 Suppression des scripts inutiles et doublons..."

# 🗑 Suppression des scripts de test obsolètes
rm -f scripts/test_model_v3.py
rm -f scripts/test_model_v6.py
rm -f scripts/test_model_v7.py

# 🗑 Suppression des anciens scripts d'entraînement
rm -f scripts/train_audio_image_model_v2.py
rm -f scripts/train_audio_image_model_v4.py
rm -f scripts/train_audio_image_model_v6.py

# 🗑 Suppression des scripts qui créent des fichiers mal formés
rm -f scripts/generate_mapping.py
rm -f scripts/update_test_mapping.py

# 🗑 Suppression des scripts de test en double
rm -f scripts/test_2en1.py
rm -f scripts/test_asso.py
rm -f scripts/test_audio_image_model.py

# 🗑 Suppression des scripts de vérification qui ne servent plus
rm -f scripts/verify_test_data.py
rm -f scripts/verify_train_test_split.py
rm -f scripts/check_arborescence.py

# 🗑 Suppression des anciens scripts d’extraction
rm -f scripts/extract_audio.py
rm -f scripts/extract_images.py

echo "✅ Suppression terminée !"
