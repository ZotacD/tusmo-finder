from dotenv import load_dotenv
import os
import time
from mistralai import Mistral
from pathlib import Path
from PIL import Image
import pytesseract

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GRID_ROWS_TO_SCAN = int(os.getenv("GRID_ROWS_TO_SCAN"))
GRIDS_IMAGES_FOLDER = os.getenv("GRIDS_IMAGES_FOLDER")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
TESSERACT_CROP_MARGIN = float(os.getenv("TESSERACT_CROP_MARGIN"))
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------- UTILITAIRE COULEURS ----------

def parse_rgb(color):
    """Convertit une couleur ('#RRGGBB' ou 'r g b') en tuple (r,g,b)."""
    if isinstance(color, tuple):
        return color
    if isinstance(color, str):
        c = color.strip()
        if c.startswith("#") and len(c) == 7:
            return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        parts = c.split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return tuple(map(int, parts))
    return (0, 0, 0)

def is_red(rgb):
    r, g, b = rgb
    return r > 150 and g < 100 and b < 100

def is_yellow(rgb):
    r, g, b = rgb
    return r > 180 and g > 150 and b < 140

def is_whiteish(rgb):
    r, g, b = rgb
    return r >= 250 and g >= 250 and b >= 250

def color_label_from_rgb(rgb):
    if is_red(rgb):
        return "red"
    if is_yellow(rgb):
        return "yellow"
    return "none"

# ---------- DÉTECTION AUTOMATIQUE DE LA GRILLE ----------

def detect_grid(img):
    """
    Détecte automatiquement le rectangle englobant la grille blanche.
    Retourne (x0, y0, grid_w et grid_h) en coordonnées pixels, avec le nombre de colonnes.

    - On scanne depuis chaque bord (haut, bas, gauche, droite).
    - Dès qu'on rencontre la "frontière" de pixels blancs, on s'arrête
    - La grid est donc définie par ces 4 premières frontières rencontrées.
    """
        
    width, height = img.size
    px = img.load()

    x_min, x_max = 0, width
    y_min, y_max = 0, height
    white_count = 0

    # --- Bord haut ---
    for y in range(height):
        for x in range(width):
            if is_whiteish(px[x, y]):
                y_min = y
                white_count += 1
                break
        if y_min != 0:
            break

    # --- Bord bas ---
    for y in range(height - 1, -1, -1):
        for x in range(width):
            if is_whiteish(px[x, y]):
                y_max = y
                white_count += 1
                break
        if y_max != height:
            break

    # --- Bord gauche ---
    for x in range(width):
        for y in range(height):
            if is_whiteish(px[x, y]):
                x_min = x
                white_count += 1
                break
        if x_min != 0:
            break

    # --- Bord droit ---
    for x in range(width - 1, -1, -1):
        for y in range(height):
            if is_whiteish(px[x, y]):
                x_max = x
                white_count += 1
                break
        if x_max != width:
            break

    if white_count == 0 or x_max <= x_min or y_max <= y_min:
        raise ValueError('Impossible de détecter clairement la grille blanche')

    tile = img.crop((x_min, y_min, x_max, y_max))
    tile.save(f".\\tesseract_crops\\grid.png")

    grid_w = x_max - x_min
    grid_h = y_max - y_min

    cell_h = grid_h / GRID_ROWS_TO_SCAN
    raw_cols = grid_w / cell_h
    nb_cols = int(round(raw_cols))

    return x_min, y_min, x_max, y_max, grid_w, grid_h, nb_cols

# ---------- LECTURE IMAGE / OCR ----------

def ocr_letter_from_tile(img, x0, y0, x1, y1):
    """
    Utilise Tesseract pour lire la lettre dans la case (x0,y0,x1,y1).
    """

    tile = img.crop((x0, y0, x1, y1))
    tile_gray = tile.convert("P")

    config = (
        "--psm 10 --oem 3 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        "-c tessedit_char_blacklist=0123456789"
    )

    text = pytesseract.image_to_string(
        tile_gray,
        config=config
    )
    
    tile_gray.save(f".\\tesseract_crops\\tile_{x0}_{y0}_{x1}_{y1}.png")

    text = text.strip().upper().replace("|", "I")
    if not text:
        return ""

    return text[0]


def scan_grid_tiles(image_path):
    """
    Parcourt toute la grille et retourne :
    - tiles : liste des cases avec leurs détails
    - nb_letter : nombre de colonnes détecté
    """
    img = Image.open(image_path).convert("RGB")

    # --- Détection automatique de la grille et du nombre de colonnes ---
    gx0, gy0, gy2, gy1, grid_w, grid_h, nb_letter = detect_grid(img)

    tiles = []

    for row in range(GRID_ROWS_TO_SCAN):
        for col in range(nb_letter):
            cell_w = (grid_w / nb_letter)
            cell_h = (grid_h / GRID_ROWS_TO_SCAN)

            x0 = int(gx0 + cell_w * col)
            y0 = int(gy0 + cell_h * row)
            x1 = int(x0 + cell_w)
            y1 = int(y0 + cell_h)

            x0c = int(x0 + cell_w * TESSERACT_CROP_MARGIN)
            y0c = int(y0 + cell_h * TESSERACT_CROP_MARGIN)
            x1c = int(x1 - cell_w * TESSERACT_CROP_MARGIN)
            y1c = int(y1 - cell_h * TESSERACT_CROP_MARGIN)

            # Point échantillon pour la couleur de la case
            cx = int(x1 - (cell_w * 0.05)) # 95% de la case
            cy = int(y0 + (cell_h * 0.5)) # milieu vertical de la case

            rgb = img.getpixel((cx, cy))
            color_label = color_label_from_rgb(rgb)

            letter = ocr_letter_from_tile(img, x0c, y0c, x1c, y1c)

            tiles.append(
                {
                    "row": row,
                    "col": col,
                    "rgb": rgb,
                    "color": color_label,
                    "letter": letter,
                }
            )

    return tiles, nb_letter


def has_colored_tiles(tiles):
    return any(t["color"] in ("red", "yellow") for t in tiles)

# ---------- CONSTRUCTION DES CONTRAINTES POUR MISTRAL ----------

def build_words_by_row(tiles, nb_letter):
    """Reconstruit les mots présents sur chaque ligne."""

    rows = {}
    for t in tiles:
        row = t["row"]
        col = t["col"]
        letter = t["letter"] if t["letter"] else "."
        rows.setdefault(row, ["." for _ in range(nb_letter)])
        rows[row][col] = letter
    # transforme en chaîne, en supprimant les points de fin inutiles
    words = {}
    for row, letters in rows.items():
        word = "".join(letters)
        word = word.rstrip(".")  # enlève les trous à droite
        words[row] = word
    return words


def detect_starting_letter(tiles):
    """Trouve la première lettre visible (haut/gauche) de la grille."""
    sorted_tiles = sorted(tiles, key=lambda t: (t["row"], t["col"]))
    for t in sorted_tiles:
        if t["letter"]:
            return t["letter"]
    return None


def build_constraints_from_tiles(tiles, nb_letter):
    """
    À partir des cases rouges/jaunes, construit un texte décrivant :
      - les lettres bien placées (rouge),
      - les lettres présentes mais mal placées (jaune) et les positions interdites,
      - les mots de chaque ligne.
    """
    correct_positions = {}
    wrong_positions = {}
    attempts_by_row = {}

    for t in tiles:
        row = t["row"]
        col = t["col"]
        color = t["color"]
        letter = t["letter"]

        if not letter or color not in ("red", "yellow"):
            continue

        attempts_by_row.setdefault(row, []).append((col, letter, color))

        if color == "red":
            correct_positions[col] = letter
        elif color == "yellow":
            wrong_positions.setdefault(letter, set()).add(col)

    lines = []
    lines.append(f"Le mot à trouver contient {nb_letter} lettres.")

    if correct_positions:
        lignes = ", ".join(
            f"position {c + 1} = {l}" for c, l in sorted(correct_positions.items())
        )
        lines.append("Lettres bien placées (cases rouges) : " + lignes + ".")

    if wrong_positions:
        parts = []
        for letter, cols in sorted(wrong_positions.items()):
            cols_txt = ", ".join(str(c + 1) for c in sorted(cols))
            parts.append(
                f"{letter} est dans le mot mais pas en position(s) {cols_txt}"
            )
        lines.append(
            "Lettres présentes mais mal placées (cases jaunes) : "
            + "; ".join(parts)
            + "\n"
        )

    if attempts_by_row:
        lines.append("Détail par tentative (chaque ligne est un mot joué) :")
        for row, entries in sorted(attempts_by_row.items()):
            descs = []
            for col, letter, color in sorted(entries):
                if color == "red":
                    descs.append(f"{letter} rouge en position {col + 1}")
                else:
                    descs.append(f"{letter} jaune en position {col + 1}")
            lines.append(f"  - Ligne {row + 1} : " + ", ".join(descs) + ".")

    words_by_row = build_words_by_row(tiles, nb_letter)
    if words_by_row:
        lines.append("Mots visibles ligne par ligne (points = cases vides) :")
        for row, word in sorted(words_by_row.items()):
            if word:  # évite les lignes complètement vides
                lines.append(f"  - Ligne {row + 1} : {word}")

    return "\n".join(lines)

def build_prompt(tiles, nb_letter):
    grid_has_colors = has_colored_tiles(tiles)
    constraints_text = build_constraints_from_tiles(tiles, nb_letter)
    starting_letter = detect_starting_letter(tiles)

    if grid_has_colors:
        prompt = (
            "Tu joues au jeu TUSMO (type Wordle français).\n"
            "Tu ne vois pas la grille, mais tu connais les informations suivantes :\n\n"
            f"{constraints_text}\n\n"
            "En respectant STRICTEMENT ces contraintes, propose un mot français possible "
            f"de {nb_letter} lettres. Ce mot doit existe dans le dictionnaire larousse. Réponds UNIQUEMENT par le mot en MAJUSCULES, "
            "sans autre commentaire."
        )
    else:
        if starting_letter:
            prompt = (
                "Tu joues au jeu TUSMO (type Wordle français).\n"
                "Aucune case rouge ou jaune n'a encore été révélée, "
                "mais on voit la première lettre du mot.\n"
                f"Le mot à trouver contient {nb_letter} lettres et commence par la lettre "
                f"{starting_letter}.\n"
                f"Propose un mot français correspondant qui existe dans le dictionnaire. Réponds UNIQUEMENT par le mot "
                "en MAJUSCULES, sans commentaire."
            )
        else:
            prompt = (
                "Tu joues au jeu TUSMO (type Wordle français).\n"
                "Aucune case rouge ou jaune n'a encore été révélée et aucune lettre "
                "n'est clairement lisible.\n"
                f"Propose simplement un mot français de {nb_letter} lettres qui existe dans le dictionnaire "
                "en MAJUSCULES. Réponds UNIQUEMENT par le mot, sans commentaire."
            )

    return prompt

# ---------- APPEL MISTRAL ----------

def call_mistral(prompt):
    client = Mistral(api_key=MISTRAL_API_KEY)

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=20,
        temperature=0.05,
    )

    content = response.choices[0].message.content
    return content.strip().split()[0]

# ---------- GESTION DES FICHIERS D’IMAGE ----------

def clear_folder(path):
    for nom in os.listdir(path):
        chemin = os.path.join(path, nom)
        if os.path.isfile(chemin) or os.path.islink(chemin):
            os.remove(chemin)
        elif os.path.isdir(chemin):
            clear_folder(chemin)
            os.rmdir(chemin)

def get_latest_image(folder_path):
    folder = Path(folder_path)
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in exts and f.is_file()]

    if not images:
        return None

    latest_image = max(images, key=lambda f: f.stat().st_mtime)
    return latest_image

# ---------- TRAITEMENT D'UNE GRILLE ----------

def process_grid(image_path):
    clear_folder('./tesseract_crops')

    print("\nImage analysée :", image_path)

    print("Détection automatique de la grille + scan détaillé...")
    tiles, nb_letter = scan_grid_tiles(str(image_path))

    print(f"\nNombre de colonnes détecté : {nb_letter}")

    # Affichage des cases rouges/jaunes + lettre associée
    print("\nCases colorées détectées (rouge/jaune) :")
    colored_tiles = [
        t for t in tiles if t["color"] in ("red", "yellow") and t["letter"]
    ]
    if not colored_tiles:
        print("  Aucune (ou OCR n'a pas réussi à lire les lettres).")
    else:
        for t in colored_tiles:
            print(
                f"  Ligne {t['row']+1}, Col {t['col']+1} : "
                f"{t['letter']} ({t['color']}, rgb={t['rgb']})"
            )

    has_colors_flag = has_colored_tiles(tiles)
    if has_colors_flag:
        print("\n→ Essais détectés (rouge/jaune présents).")
    else:
        print("\n→ Grille sans couleurs (début de partie ou essais infructueux).")

    print("\nContraintes construites pour Mistral :")
    print(build_constraints_from_tiles(tiles, nb_letter))

    prompt = build_prompt(tiles, nb_letter)
    print("\nPrompt:", prompt)

    print("\nAppel à Mistral AI...")
    guess = call_mistral(prompt)

    print("\nMot proposé :", guess)

# ---------- MAIN ----------

def main():
    image_path = get_latest_image(GRIDS_IMAGES_FOLDER)
    process_grid(image_path)

if __name__ == "__main__":
    main()
