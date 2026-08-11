"""Compute PSNR/SSIM between CycleGAN test.py outputs and ground-truth test images.

Because datasets/maps/testA and testB share the same numeric index per file
(e.g. 1000_A.jpg / 1000_B.jpg) and test.py was run with --serial_batches,
fake_B images correspond index-for-index to real testB images (and vice versa
for fake_A / testA). We use that to get a rough quantitative translation
quality score, plus cycle-reconstruction quality (rec_A vs real_A, rec_B vs real_B).
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

IMAGES_DIR = Path(sys.argv[1])
DATAROOT = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])

testA_dir = DATAROOT / "testA"
testB_dir = DATAROOT / "testB"


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def index_of(fname):
    m = re.match(r"(\d+)_", fname)
    return m.group(1) if m else None


rows = []
fakeB_files = sorted(IMAGES_DIR.glob("*_fake_B.png"))
fakeA_files = sorted(IMAGES_DIR.glob("*_fake_A.png"))
recA_files = {index_of(p.name): p for p in IMAGES_DIR.glob("*_rec_A.png")}
recB_files = {index_of(p.name): p for p in IMAGES_DIR.glob("*_rec_B.png")}
realA_files = {index_of(p.name): p for p in IMAGES_DIR.glob("*_real_A.png")}
realB_files = {index_of(p.name): p for p in IMAGES_DIR.glob("*_real_B.png")}

print(f"Found {len(fakeB_files)} fake_B, {len(fakeA_files)} fake_A images in {IMAGES_DIR}")

for fb in fakeB_files:
    idx = index_of(fb.name)
    gt_path = testB_dir / f"{idx}_B.jpg"
    rec_path = recA_files.get(idx)
    if not gt_path.exists():
        continue
    fake_b = load(fb)
    gt_b = load(gt_path)
    if fake_b.shape != gt_b.shape:
        gt_b = np.array(Image.fromarray(gt_b).resize(fake_b.shape[1::-1]))
    p = psnr(gt_b, fake_b, data_range=255)
    s = ssim(gt_b, fake_b, data_range=255, channel_axis=2)

    rec_psnr = rec_ssim = None
    real_a = realA_files.get(idx)
    if rec_path and real_a:
        ra = load(real_a)
        rc = load(rec_path)
        if ra.shape == rc.shape:
            rec_psnr = psnr(ra, rc, data_range=255)
            rec_ssim = ssim(ra, rc, data_range=255, channel_axis=2)

    rows.append({
        "index": idx,
        "direction": "A_to_B",
        "psnr_vs_groundtruth": p,
        "ssim_vs_groundtruth": s,
        "cycle_rec_psnr": rec_psnr,
        "cycle_rec_ssim": rec_ssim,
    })

for fa in fakeA_files:
    idx = index_of(fa.name)
    gt_path = testA_dir / f"{idx}_A.jpg"
    rec_path = recB_files.get(idx)
    if not gt_path.exists():
        continue
    fake_a = load(fa)
    gt_a = load(gt_path)
    if fake_a.shape != gt_a.shape:
        gt_a = np.array(Image.fromarray(gt_a).resize(fake_a.shape[1::-1]))
    p = psnr(gt_a, fake_a, data_range=255)
    s = ssim(gt_a, fake_a, data_range=255, channel_axis=2)

    rec_psnr = rec_ssim = None
    real_b = realB_files.get(idx)
    if rec_path and real_b:
        rb = load(real_b)
        rc = load(rec_path)
        if rb.shape == rc.shape:
            rec_psnr = psnr(rb, rc, data_range=255)
            rec_ssim = ssim(rb, rc, data_range=255, channel_axis=2)

    rows.append({
        "index": idx,
        "direction": "B_to_A",
        "psnr_vs_groundtruth": p,
        "ssim_vs_groundtruth": s,
        "cycle_rec_psnr": rec_psnr,
        "cycle_rec_ssim": rec_ssim,
    })

OUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUT_DIR / "per_image_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary_lines = ["# CycleGAN maps_cyclegan_slurm — test set metrics\n"]
for direction in ["A_to_B", "B_to_A"]:
    sub = [r for r in rows if r["direction"] == direction]
    if not sub:
        continue
    psnrs = np.array([r["psnr_vs_groundtruth"] for r in sub])
    ssims = np.array([r["ssim_vs_groundtruth"] for r in sub])
    rec_psnrs = np.array([r["cycle_rec_psnr"] for r in sub if r["cycle_rec_psnr"] is not None])
    rec_ssims = np.array([r["cycle_rec_ssim"] for r in sub if r["cycle_rec_ssim"] is not None])
    summary_lines.append(f"\n## {direction}  (n={len(sub)})\n")
    summary_lines.append(f"- PSNR vs ground truth: mean={psnrs.mean():.3f}  std={psnrs.std():.3f}\n")
    summary_lines.append(f"- SSIM vs ground truth: mean={ssims.mean():.4f}  std={ssims.std():.4f}\n")
    if len(rec_psnrs):
        summary_lines.append(f"- Cycle-reconstruction PSNR: mean={rec_psnrs.mean():.3f}  std={rec_psnrs.std():.3f}\n")
        summary_lines.append(f"- Cycle-reconstruction SSIM: mean={rec_ssims.mean():.4f}  std={rec_ssims.std():.4f}\n")

with open(OUT_DIR / "metrics_summary.md", "w") as f:
    f.writelines(summary_lines)

print("".join(summary_lines))
print(f"\nWrote {csv_path} and {OUT_DIR / 'metrics_summary.md'}")
