import os
import random
import io
from PIL import Image, ImageFilter, ImageOps
import cairosvg
import numpy as np

# ============================================================
# Settings
# ============================================================
OUTPUT_DIR = "synthetic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = [
    "empty",
    "wP","wN","wB","wR","wQ","wK",
    "bP","bN","bB","bR","bQ","bK"
]

# chess.com–like square colors
LIGHT_SQUARES = [(240, 217, 181), (235, 210, 173)]
DARK_SQUARES  = [(181, 136, 99), (175, 130, 92)]

PIECE_SCALE_RANGE = (0.55, 0.75)  # Proportion of square
SQUARE_SIZE = 224

# ---------------------------
# TEST = 5 per class
# FULL TRAIN = 100,000 per class
# ---------------------------
IMAGES_PER_CLASS = 5  # ← Change to 100000 when ready


# ============================================================
# Render SVG → PNG
# ============================================================
def render_svg(svg_path, size):
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


# ============================================================
# Make empty square
# ============================================================
def make_empty_square():
    base_color = random.choice(LIGHT_SQUARES + DARK_SQUARES)
    img = Image.new("RGB", (SQUARE_SIZE, SQUARE_SIZE), base_color)

    # Add subtle texture (prevents overfitting)
    noise = (np.random.rand(SQUARE_SIZE, SQUARE_SIZE, 1) * 8).astype("uint8")
    noise = np.repeat(noise, 3, axis=2)
    img = Image.fromarray(np.clip(np.array(img) + noise, 0, 255).astype("uint8"))

    return img


# ============================================================
# Paste piece on square
# ============================================================
def paste_piece(square, piece_img):
    sq = square.copy()

    scale = random.uniform(*PIECE_SCALE_RANGE)
    new_size = int(SQUARE_SIZE * scale)

    piece_resized = piece_img.resize((new_size, new_size), Image.LANCZOS)

    x = (SQUARE_SIZE - new_size) // 2
    y = (SQUARE_SIZE - new_size) // 2

    sq.paste(piece_resized, (x, y), piece_resized)
    return sq


# ============================================================
# Image augmentation
# ============================================================
def augment(img):
    # Small blur variation
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    # Slight brightness variance
    arr = np.array(img).astype("float32")
    arr *= random.uniform(0.92, 1.08)
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype("uint8"))

    return img


# ============================================================
# Main Generator
# ============================================================
def generate_synthetic_dataset(svg_folder="pieces"):
    for cls in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Generating: {cls} ({IMAGES_PER_CLASS} images)")

        # ----------------------------------------------------
        # EMPTY CLASS
        # ----------------------------------------------------
        if cls == "empty":
            for i in range(IMAGES_PER_CLASS):
                img = augment(make_empty_square())
                img.save(os.path.join(class_dir, f"{cls}_{i}.png"))
            continue

        # ----------------------------------------------------
        # PIECE CLASS
        # ----------------------------------------------------
        svg_file = os.path.join(svg_folder, f"{cls}.svg")
        if not os.path.exists(svg_file):
            print(f"[WARN] Missing SVG for class: {cls}")
            continue

        # load once per class
        piece_png = render_svg(svg_file, 512)

        for i in range(IMAGES_PER_CLASS):
            square = make_empty_square()
            comp = paste_piece(square, piece_png)
            comp = augment(comp)
            comp.save(os.path.join(class_dir, f"{cls}_{i}.png"))


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    generate_synthetic_dataset()
