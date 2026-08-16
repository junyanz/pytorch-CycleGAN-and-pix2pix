"""For every `videos`-task run (any subject, any session, any run number) under
--runs-root, apply the age-appropriate fcgmask, take the median volume across time,
and compute:
  - how many axial/coronal/sagittal slices are entirely empty (no brain voxels) after masking
  - a bounding box around the brain content (tight, and padded to a multiple of 4 -- the
    padding a 3D CNN with 2 downsampling stages needs to reconstruct the exact input shape)
  - the effective (cropped) volume size and voxel count

Writes one row per run to --out-csv, incrementally (flushed after every run), so a partial
result is preserved even if the job times out partway through. Designed to run standalone or
under SLURM via submit_bbox_job.sbatch in this same directory.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

MASK_PATH = Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz")

RUN_DIR_RE = re.compile(r"_run_(\d+)_session_(\d+)_task_name_(\w+)")
SUBJECT_DIR_RE = re.compile(r"_subject_id_(\w+)")

CSV_FIELDS = [
    "subject_id", "session_id", "run_id", "task", "run_path",
    "orig_shape_x", "orig_shape_y", "orig_shape_z", "n_volumes",
    "n_empty_slices_x", "n_empty_slices_y", "n_empty_slices_z",
    "bbox_x_lo", "bbox_x_hi", "bbox_y_lo", "bbox_y_hi", "bbox_z_lo", "bbox_z_hi",
    "effective_shape_x", "effective_shape_y", "effective_shape_z", "effective_volume_voxels",
    "bbox_pad4_x_lo", "bbox_pad4_x_hi", "bbox_pad4_y_lo", "bbox_pad4_y_hi", "bbox_pad4_z_lo", "bbox_pad4_z_hi",
    "padded_shape_x", "padded_shape_y", "padded_shape_z",
    "status", "message",
]


def find_video_runs(runs_root):
    """Every videos-task run file under runs_root, any subject/session/run number."""
    return sorted(runs_root.glob("_subject_id_*/_referencetype_standard/_run_*_session_*_task_name_videos/*_bold_mcf_corrected_flirt.nii.gz"))


def parse_run_identity(run_path):
    subject_match = SUBJECT_DIR_RE.search(run_path.parts[-4])
    run_dir_match = RUN_DIR_RE.search(run_path.parts[-2])
    subject_id = subject_match.group(1) if subject_match else None
    run_id, session_id, task = (run_dir_match.groups() if run_dir_match else (None, None, None))
    return subject_id, session_id, run_id, task


def axis_bbox(mask3d, axis):
    """(lo, hi) of nonzero slices along `axis` (hi exclusive), and count of empty slices."""
    other = tuple(a for a in range(3) if a != axis)
    nz = np.where(mask3d.any(axis=other))[0]
    n_total = mask3d.shape[axis]
    if len(nz) == 0:
        return None, None, n_total
    lo, hi = int(nz.min()), int(nz.max()) + 1
    n_empty = n_total - (hi - lo)
    return lo, hi, n_empty


def pad_bbox(lo, hi, total, pad_to_multiple=4):
    pad = (-(hi - lo)) % pad_to_multiple
    lo2 = max(0, lo - pad // 2)
    hi2 = min(total, hi + (pad - pad // 2))
    return lo2, hi2


def compute_run_row(run_path, mask):
    subject_id, session_id, run_id, task = parse_run_identity(run_path)
    img = nib.load(run_path)
    data = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
    orig_shape = data.shape[:3]
    n_volumes = data.shape[-1]

    if mask.shape != orig_shape:
        raise ValueError(f"mask shape {mask.shape} does not match run shape {orig_shape}")

    masked = data * mask[..., None]
    median_vol = np.median(masked, axis=-1)
    brain_mask_3d = median_vol > 0

    bbox, empties = {}, {}
    for axis, name in enumerate("xyz"):
        lo, hi, n_empty = axis_bbox(brain_mask_3d, axis)
        if lo is None:
            raise ValueError(f"no brain voxels found along axis {name} -- mask/data mismatch?")
        bbox[name] = (lo, hi)
        empties[name] = n_empty

    padded = {name: pad_bbox(lo, hi, orig_shape[axis], pad_to_multiple=4) for axis, (name, (lo, hi)) in enumerate(bbox.items())}

    effective_shape = {name: hi - lo for name, (lo, hi) in bbox.items()}
    padded_shape = {name: hi - lo for name, (lo, hi) in padded.items()}
    effective_voxels = effective_shape["x"] * effective_shape["y"] * effective_shape["z"]

    return {
        "subject_id": subject_id, "session_id": session_id, "run_id": run_id, "task": task,
        "run_path": str(run_path),
        "orig_shape_x": orig_shape[0], "orig_shape_y": orig_shape[1], "orig_shape_z": orig_shape[2],
        "n_volumes": n_volumes,
        "n_empty_slices_x": empties["x"], "n_empty_slices_y": empties["y"], "n_empty_slices_z": empties["z"],
        "bbox_x_lo": bbox["x"][0], "bbox_x_hi": bbox["x"][1],
        "bbox_y_lo": bbox["y"][0], "bbox_y_hi": bbox["y"][1],
        "bbox_z_lo": bbox["z"][0], "bbox_z_hi": bbox["z"][1],
        "effective_shape_x": effective_shape["x"], "effective_shape_y": effective_shape["y"], "effective_shape_z": effective_shape["z"],
        "effective_volume_voxels": effective_voxels,
        "bbox_pad4_x_lo": padded["x"][0], "bbox_pad4_x_hi": padded["x"][1],
        "bbox_pad4_y_lo": padded["y"][0], "bbox_pad4_y_hi": padded["y"][1],
        "bbox_pad4_z_lo": padded["z"][0], "bbox_pad4_z_hi": padded["z"][1],
        "padded_shape_x": padded_shape["x"], "padded_shape_y": padded_shape["y"], "padded_shape_z": padded_shape["z"],
        "status": "ok", "message": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/normalized_to_common_space"))
    parser.add_argument("--mask-path", type=Path, default=MASK_PATH)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N runs (for testing)")
    args = parser.parse_args()

    mask = nib.load(args.mask_path).get_fdata() > 0

    run_files = find_video_runs(args.runs_root)
    if args.limit:
        run_files = run_files[: args.limit]
    print(f"found {len(run_files)} videos-task runs under {args.runs_root}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    t_start = time.time()
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        for i, run_path in enumerate(run_files):
            t0 = time.time()
            try:
                row = compute_run_row(run_path, mask)
                n_ok += 1
            except Exception as e:
                subject_id, session_id, run_id, task = parse_run_identity(run_path)
                row = {field: "" for field in CSV_FIELDS}
                row.update(subject_id=subject_id, session_id=session_id, run_id=run_id, task=task,
                           run_path=str(run_path), status="failed", message=str(e))
                n_failed += 1
                print(f"  [{i+1}/{len(run_files)}] FAILED {run_path.name}: {e}", flush=True)
            writer.writerow(row)
            f.flush()
            elapsed = time.time() - t0
            if row["status"] == "ok":
                print(f"  [{i+1}/{len(run_files)}] {row['subject_id']} ses-{row['session_id']} run-{row['run_id']}: "
                      f"bbox=({row['effective_shape_x']},{row['effective_shape_y']},{row['effective_shape_z']}) "
                      f"in {elapsed:.1f}s", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\ndone: {n_ok} ok, {n_failed} failed, {total_elapsed/60:.1f} min total", flush=True)
    print(f"saved to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
