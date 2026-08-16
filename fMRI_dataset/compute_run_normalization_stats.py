"""Precompute robust_p5p95 normalization stats (median, scale) once per source run, for every
run referenced in chunk_metadata.csv. The dataloader (fmri_chunk_dataset.py) reads the resulting
CSV and looks up a chunk's run's (median, scale) by (subject_id, session_id, run_id, task) --
computing stats per-chunk instead would be both slow (reloading full runs repeatedly) and
statistically noisy (5 timepoints is too few for a stable percentile estimate; see the tSNR/DVARS
caveat in the sanity-check script and normalization_study.ipynb for the same reasoning applied
to normalization).

Normalization: (x - median) / (p95 - p5), computed from brain-mask voxels only (mask = nonzero at
t=0 -- equivalent to the source fcgmask, verified in normalization_study.ipynb). Unclipped -- see
normalization_study.ipynb's "Chosen strategy" section for why (the project's residual generator
isn't Tanh-bounded, so there's no need to clip, and this keeps normalization exactly invertible).

Usage:
    python compute_run_normalization_stats.py \
        --chunk-metadata-csv ../data_preprocessing/motion_grades_chunk_5_dataset/chunk_metadata.csv \
        --out-csv run_normalization_stats.csv
"""

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNK_METADATA_CSV = THIS_DIR.parent / "data_preprocessing/motion_grades_chunk_5_dataset/chunk_metadata.csv"
DEFAULT_OUT_CSV = THIS_DIR / "run_normalization_stats.csv"

STATS_CSV_FIELDS = ["subject_id", "session_id", "run_id", "task", "median", "scale", "n_brain_voxels", "source_volume_path"]


def unique_runs(chunk_metadata_csv):
    """One row per unique (subject_id, session_id, run_id, task), keeping its source_volume_path."""
    seen = {}
    with open(chunk_metadata_csv) as f:
        for row in csv.DictReader(f):
            key = (row["subject_id"], row["session_id"], row["run_id"], row["task"])
            seen[key] = row["source_volume_path"]
    return seen


def compute_stats_for_run(key_and_path):
    key, source_volume_path = key_and_path
    subject_id, session_id, run_id, task = key
    t0 = time.time()
    img = nib.load(source_volume_path)
    data = np.asarray(img.dataobj, dtype=np.float32)  # (H, W, D, T), already brain-masked+cropped
    mask = data[..., 0] != 0  # brain footprint is identical across all T timepoints (masked upstream)

    x_brain = data[mask]
    median = float(np.median(x_brain))
    p5, p95 = np.percentile(x_brain, [5, 95])
    scale = float(max(p95 - p5, 1e-8))

    return dict(subject_id=subject_id, session_id=session_id, run_id=run_id, task=task,
                median=median, scale=scale, n_brain_voxels=int(mask.sum()),
                source_volume_path=source_volume_path, elapsed_s=round(time.time() - t0, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-metadata-csv", type=Path, default=DEFAULT_CHUNK_METADATA_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    runs = unique_runs(args.chunk_metadata_csv)
    print(f"found {len(runs)} unique runs referenced in {args.chunk_metadata_csv}", flush=True)

    t_start = time.time()
    n_ok = n_failed = 0
    with open(args.out_csv, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=STATS_CSV_FIELDS)
        writer.writeheader()
        futures = {pool.submit(compute_stats_for_run, item): item for item in runs.items()}
        for i, fut in enumerate(as_completed(futures)):
            key, path = futures[fut]
            try:
                row = fut.result()
                writer.writerow({k: row[k] for k in STATS_CSV_FIELDS})
                f.flush()
                n_ok += 1
                if (i + 1) % 50 == 0 or (i + 1) == len(runs):
                    print(f"  [{i+1}/{len(runs)}] {key}: median={row['median']:.2f} scale={row['scale']:.2f} "
                          f"({row['elapsed_s']}s)", flush=True)
            except Exception as e:
                n_failed += 1
                print(f"  [{i+1}/{len(runs)}] FAILED {key} ({path}): {e}", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\ndone in {total_elapsed/60:.1f} min: {n_ok} ok, {n_failed} failed", flush=True)
    print(f"saved to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
