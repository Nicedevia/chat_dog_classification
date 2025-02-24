import os
import cv2
import numpy as np

IMAGE_SIZE = (64, 64)
DATA_PATH = "data/extracted"

def preprocess_images(input_folder):
    for category in ["cats", "dogs"]:
        folder_path = os.path.join(input_folder, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE) / 255.0
                cv2.imwrite(img_path, (img * 255).astype(np.uint8))

print("🔄 Prétraitement des images...")
preprocess_images(os.path.join(DATA_PATH, "training_set"))
preprocess_images(os.path.join(DATA_PATH, "test_set"))
print("✅ Prétraitement des images terminé !")
