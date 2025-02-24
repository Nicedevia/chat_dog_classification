import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import random

# 📂 Dossiers source et destination
source_dir = "data/audio/cleaned"
dest_dir = "data/audio/augmented"

# 🏗 Création des dossiers
os.makedirs(os.path.join(dest_dir, "cats"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "dogs"), exist_ok=True)

# 🎛 Paramètres audio
SR = 22050  # Fréquence d'échantillonnage

# 🔹 Fonctions d'augmentation
def add_white_noise(y, noise_level=0.02):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate=1.2):
    return librosa.effects.time_stretch(y, rate=float(rate))

def reverse_audio(y):
    return np.flip(y)

# 🔄 Fonction principale d'augmentation
def augment_audio(file_path, output_folder, augment_factor=3):
    y, sr = librosa.load(file_path, sr=SR)

    for i in range(augment_factor):
        aug_y = np.copy(y)

        transformation = random.choice(["noise", "pitch", "speed", "reverse"])
        if transformation == "noise":
            aug_y = add_white_noise(aug_y)
        elif transformation == "pitch":
            aug_y = pitch_shift(aug_y, sr, n_steps=random.choice([-3, -2, 2, 3]))
        elif transformation == "speed":
            aug_y = time_stretch(aug_y, rate=random.choice([0.8, 1.2]))
        elif transformation == "reverse":
            aug_y = reverse_audio(aug_y)

        # 🔹 Normalisation du volume
        aug_y = librosa.util.normalize(aug_y)

        # 💾 Sauvegarde
        filename = os.path.basename(file_path).replace(".wav", f"_aug_{i}.wav")
        output_path = os.path.join(output_folder, filename)

        sf.write(output_path, aug_y, sr)
        print(f"✅ Fichier généré : {output_path}")

# 🏗 Appliquer l'augmentation
for category in ["cats", "dogs"]:
    input_folder = os.path.join(source_dir, category)
    output_folder = os.path.join(dest_dir, category)

    augment_factor = 3 if category == "dogs" else 2  # 🎯 Augmente plus les chiens

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(input_folder, file)
            augment_audio(file_path, output_folder, augment_factor)

print("\n🎵 Augmentation des données audio terminée ! ✅")
