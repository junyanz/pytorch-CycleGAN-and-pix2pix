"""Build side-by-side comparison grids from CycleGAN test.py outputs."""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

IMAGES_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
N_SAMPLES = 8
random.seed(0)

fake_b_files = sorted(IMAGES_DIR.glob("*_fake_B.png"))
sample = random.sample(fake_b_files, N_SAMPLES)

PAD = 6
LABEL_H = 22
CELL = 256


def label(draw, x, y, text):
    draw.text((x + 4, y + 3), text, fill=(255, 255, 0))


def build_grid(rows, col_labels, out_path):
    n_rows = len(rows)
    n_cols = len(col_labels)
    w = n_cols * CELL + (n_cols + 1) * PAD
    h = n_rows * (CELL + LABEL_H) + (n_rows + 1) * PAD
    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for r, imgs in enumerate(rows):
        for c, im_path in enumerate(imgs):
            im = Image.open(im_path).convert("RGB").resize((CELL, CELL))
            x = PAD + c * (CELL + PAD)
            y = PAD + r * (CELL + LABEL_H + PAD)
            canvas.paste(im, (x, y + LABEL_H))
            if r == 0:
                label(draw, x, y, col_labels[c])
    canvas.save(out_path)
    print(f"saved {out_path}")


rows_a2b, rows_b2a = [], []
for fb in sample:
    stem = fb.name.replace("_fake_B.png", "")
    real_a = IMAGES_DIR / f"{stem}_real_A.png"
    fake_b = fb
    real_b = IMAGES_DIR / f"{stem}_real_B.png"
    rec_a = IMAGES_DIR / f"{stem}_rec_A.png"
    fake_a = IMAGES_DIR / f"{stem}_fake_A.png"
    rec_b = IMAGES_DIR / f"{stem}_rec_B.png"
    rows_a2b.append([real_a, fake_b, real_b, rec_a])
    rows_b2a.append([real_b, fake_a, real_a, rec_b])

OUT_DIR.mkdir(parents=True, exist_ok=True)
build_grid(rows_a2b, ["real_A (satellite)", "fake_B (generated map)", "real_B (GT map)", "rec_A (cycle back)"],
           OUT_DIR / "comparison_A_to_B.png")
build_grid(rows_b2a, ["real_B (map)", "fake_A (generated satellite)", "real_A (GT satellite)", "rec_B (cycle back)"],
           OUT_DIR / "comparison_B_to_A.png")
