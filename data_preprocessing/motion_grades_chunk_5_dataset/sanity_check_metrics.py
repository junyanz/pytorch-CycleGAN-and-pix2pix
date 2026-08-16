"""Sanity-check the motion grading by computing per-chunk QC metrics directly from the saved
5-volume NIfTI chunks, then comparing their distributions across grades. If grading is doing its
job, these metrics -- which are NOT used anywhere in the grading criteria itself -- should still
trend consistently with grade (worse image quality for higher-motion grades), which is independent
evidence the FD-based grades are capturing something real in the data, not just noise.

Metrics (all computed per-chunk, from the chunk's own 5 timepoints, within the chunk's brain mask
-- data is already brain-masked upstream, so mask = data[...,0] != 0):

  - FD (mean, mm):    pulled straight from chunk_metadata.csv (chunk_mean_fd) -- not recomputed,
                       this is the grading criterion itself, included as a positive control so the
                       other 3 "independent" metrics can be visually compared against it.
  - tSNR (median):    per-voxel temporal mean / temporal std (ddof=1) across the chunk's 5
                       timepoints, then median across brain voxels (median, not mean, since 5
                       samples makes per-voxel std noisy and a handful of near-zero-std voxels can
                       otherwise blow up the average). CAVEAT: tSNR from only 5 timepoints is a
                       very low-DOF estimate -- expect much wider spread than a typical whole-run
                       tSNR map, especially for low-motion chunks where std is naturally tiny.
  - Global signal CV (%): coefficient of variation (std/mean * 100) of the mean brain-masked
                       intensity across the chunk's 5 timepoints. Higher = the whole-brain signal
                       swings around more from volume to volume within the chunk -- a hallmark of
                       motion-induced intensity spikes.
  - DVARS (%):         RMS of the frame-to-frame voxelwise intensity difference (Power et al. 2012
                       style), scaled by the chunk's own mean masked intensity and expressed as a
                       percentage, averaged over the chunk's 4 consecutive-volume pairs.

Usage:
  python sanity_check_metrics.py --chunk-csv chunk_metadata.csv --splits test,val \
      --out-dir sanity_check_output --workers 8
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

GRADE_ORDER = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6"]
TASKS = ("videos", "rest10")

METRICS = [
    ("chunk_mean_fd", "FD mean (mm)"),
    ("tsnr_median", "tSNR (median, per-chunk)"),
    ("gs_cv_pct", "Global signal CV (%)"),
    ("dvars_pct", "DVARS (%)"),
]


def compute_chunk_metrics(row):
    """row: dict from chunk_metadata.csv (one row). Returns row augmented with tsnr_median,
    gs_cv_pct, dvars_pct, or with those set to NaN + an error message if the file couldn't be
    processed."""
    out = dict(row)
    try:
        img = nib.load(row["chunk_path"])
        data = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T=5)
        mask = data[..., 0] != 0
        voxel_ts = data[mask]  # (n_voxels, 5)
        if voxel_ts.shape[0] == 0:
            raise ValueError("empty brain mask in chunk")

        mean_t = voxel_ts.mean(axis=1)
        std_t = voxel_ts.std(axis=1, ddof=1)
        valid = std_t > 1e-6
        tsnr_median = float(np.median(mean_t[valid] / std_t[valid])) if valid.any() else float("nan")

        global_signal = voxel_ts.mean(axis=0)  # (5,)
        gs_mean = global_signal.mean()
        gs_cv_pct = float(100 * global_signal.std(ddof=1) / gs_mean) if gs_mean != 0 else float("nan")

        diffs = np.diff(voxel_ts, axis=1)  # (n_voxels, 4)
        scale = voxel_ts.mean()
        dvars_per_t = 100 * np.sqrt(np.mean(diffs ** 2, axis=0)) / scale if scale != 0 else np.full(4, np.nan)
        dvars_pct = float(dvars_per_t.mean())

        out.update(tsnr_median=tsnr_median, gs_cv_pct=gs_cv_pct, dvars_pct=dvars_pct, metric_status="ok")
    except Exception as e:
        out.update(tsnr_median=float("nan"), gs_cv_pct=float("nan"), dvars_pct=float("nan"),
                    metric_status=f"failed: {e}")
    return out


def make_boxplots(df, split, out_dir):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["split"] == split]

    for task in TASKS:
        task_sub = sub[sub["task"] == task]
        for metric_col, metric_label in METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            box_data = []
            box_labels = []
            n_per_grade = []
            for grade in GRADE_ORDER:
                vals = task_sub.loc[task_sub["grade"] == grade, metric_col].dropna().values
                box_data.append(vals)
                box_labels.append(grade.replace("Grade ", "G"))
                n_per_grade.append(len(vals))

            ax.boxplot(box_data, showfliers=True)
            xtick_labels = [f"{lbl}\n(n={n})" for lbl, n in zip(box_labels, n_per_grade)]
            ax.set_xticks(range(1, len(xtick_labels) + 1))
            ax.set_xticklabels(xtick_labels)
            ax.set_ylabel(metric_label)
            ax.set_xlabel("Grade")
            ax.set_title(f"{metric_label} by grade -- {task}, {split} split")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            fname = f"{task}_{metric_col}.png"
            fig.savefig(split_dir / fname, dpi=150)
            plt.close(fig)
    print(f"  saved plots for split={split} to {split_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-csv", type=Path, required=True)
    parser.add_argument("--splits", type=str, default="test,val", help="comma-separated splits to process")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N chunks (for testing)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    with open(args.chunk_csv) as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["split"] in splits]
    if args.limit:
        rows = rows[: args.limit]
    print(f"processing {len(rows)} chunks across splits {splits} (of {len(all_rows)} total in {args.chunk_csv})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.out_dir / "sanity_check_metrics.csv"

    results = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compute_chunk_metrics, row) for row in rows]
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            results.append(r)
            if r["metric_status"] != "ok":
                n_failed += 1
                print(f"  [{i+1}/{len(rows)}] FAILED {r['chunk_path']}: {r['metric_status']}", flush=True)
            elif (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    print(f"\ndone: {len(results)} processed, {n_failed} failed", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(metrics_csv, index=False)
    print(f"saved metrics to {metrics_csv}", flush=True)

    for col in ["chunk_mean_fd", "chunk_max_fd", "tsnr_median", "gs_cv_pct", "dvars_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for split in splits:
        if split not in df["split"].unique():
            print(f"  WARNING: no chunks found for split={split}, skipping plots", flush=True)
            continue
        make_boxplots(df, split, args.out_dir)

    print("\nsummary (median per split x task x grade):", flush=True)
    summary = df.groupby(["split", "task", "grade"])[["chunk_mean_fd", "tsnr_median", "gs_cv_pct", "dvars_pct"]].median()
    print(summary.reindex(GRADE_ORDER, level="grade"), flush=True)


if __name__ == "__main__":
    main()
