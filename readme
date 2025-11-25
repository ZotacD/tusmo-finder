# Tusmo Finder

Script Python qui lit automatiquement une grille TUSMO (type Wordle français) à partir d’une image, extrait les lettres par OCR, interprète les couleurs (rouge/jaune), puis propose un mot candidat en utilisant l’API de chat Mistral.

---

## 🧠 Technologies utilisées

### 1. OCR (Reconnaissance de texte)

- **Pillow (`PIL`)** : ouverture et manipulation des images (crop, conversion, etc.).
- **Tesseract + `pytesseract`** :
  - Détection des **lettres dans chaque case** de la grille.
  - Configuration restreinte pour ne lire que les lettres majuscules (A–Z).
  - Recadrage fin des tuiles pour améliorer la qualité de l’OCR.

### 2. Analyse d’image

- Détection automatique de la **grille blanche** :
  - Scan des bords de l’image pour trouver les premières lignes de pixels blancs.
  - Calcul de la taille de la grille et du nombre de colonnes.
- Lecture des **couleurs de cases** :
  - Analyse du pixel proche du bord droit de chaque case.
  - Classification simple en `red`, `yellow` ou `none` via des seuils RGB.

### 3. Chat / LLM (Mistral AI)

- Utilisation du client Python `mistralai`.
- Construction d’un **prompt structuré** décrivant :
  - La longueur du mot.
  - Les lettres bien placées (rouges).
  - Les lettres présentes mais mal placées (jaunes).
  - Les mots déjà joués ligne par ligne.
- Appel au modèle `mistral-large-latest` pour proposer un mot français **respectant strictement les contraintes**, réponse en MAJUSCULES.

---

## ⚙️ Prérequis

- Python 3.9+ (recommandé)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installé sur la machine
- Une clé API Mistral AI
- `pip install` des dépendances Python :
  - `python-dotenv`
  - `mistralai`
  - `pillow`
  - `pytesseract`

---

## 📁 Configuration

Le script s’appuie sur un fichier `.env` pour sa configuration :

```env
MISTRAL_API_KEY=VOTRE_CLE_API_MISTRAL
GRID_ROWS_TO_SCAN=6
GRIDS_IMAGES_FOLDER=./grids
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
TESSERACT_CROP_MARGIN=0.15
