import os

def list_directory_structure(root_dir, output_file="arborescence.txt"):
    """
    Génère l'arborescence des dossiers et fichiers sans afficher les fichiers .wav, .png et .jpg,
    mais en les comptant.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(root_dir):
            level = root.replace(root_dir, "").count(os.sep)
            indent = " " * 4 * level
            f.write(f"{indent}📂 {os.path.basename(root)}/\n")

            sub_indent = " " * 4 * (level + 1)

            # Filtrer et compter les fichiers .wav, .png, .jpg
            wav_count = sum(1 for file in files if file.endswith(".wav"))
            png_count = sum(1 for file in files if file.endswith(".png"))
            jpg_count = sum(1 for file in files if file.endswith(".jpg"))

            # Affichage des fichiers autres que .wav, .png, .jpg
            for file in files:
                if not file.endswith((".wav", ".png", ".jpg")):
                    f.write(f"{sub_indent}📄 {file}\n")

            # Ajouter les nombres de fichiers audio et images si présents
            if wav_count > 0:
                f.write(f"{sub_indent}🎵 {wav_count} fichiers .wav\n")
            if png_count > 0:
                f.write(f"{sub_indent}🖼 {png_count} fichiers .png\n")
            if jpg_count > 0:
                f.write(f"{sub_indent}📸 {jpg_count} fichiers .jpg\n")

    print(f"✅ Arborescence générée dans {output_file}")

if __name__ == "__main__":
    project_root = os.getcwd()  # Prend le dossier actuel comme racine
    list_directory_structure(project_root)
