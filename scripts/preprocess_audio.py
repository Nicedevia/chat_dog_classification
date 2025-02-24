import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

AUDIO_PATH = "data/audio/cats_dogs"
OUTPUT_PATH = "data/audio/spectrograms"

def generate_spectrograms(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for category in ["cats", "dogs"]:
        category_input = os.path.join(input_folder, category)
        category_output = os.path.join(output_folder, category)
        os.makedirs(category_output, exist_ok=True)

        for filename in os.listdir(category_input):
            if filename.endswith(".wav"):
                audio_path = os.path.join(category_input, filename)
                y, sr = librosa.load(audio_path, sr=22050)
                spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

                plt.figure(figsize=(3, 3))
                librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
                plt.axis("off")
                output_file = os.path.join(category_output, filename.replace(".wav", ".png"))
                plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
                plt.close()

print("🔄 Génération des spectrogrammes...")
generate_spectrograms(os.path.join(AUDIO_PATH, "train"), os.path.join(OUTPUT_PATH, "train"))
generate_spectrograms(os.path.join(AUDIO_PATH, "test"), os.path.join(OUTPUT_PATH, "test"))
print("✅ Génération des spectrogrammes terminée !")
