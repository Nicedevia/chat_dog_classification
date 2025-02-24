# Utilisation de l'image Python officielle
FROM python:3.9-slim

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du projet
COPY . .

# Commande pour exécuter le projet
CMD ["python", "main.py"]
