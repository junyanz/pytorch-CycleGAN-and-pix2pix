"""Build a brain-masked, cropped, uniformly-shaped dataset from the raw normalized_to_common_space
runs, mirroring the source directory structure. Supports both age groups (2mo and 9mo), each with
its own age-appropriate mask and crop window, both of which land on the exact same final shape.

Scope: tasks 'videos' and 'rest10' only (any session, any run number) -- 'pictures' runs also
exist in the source tree but are out of scope here. Age group selected via --age-group, and
determined per-subject by templates/readme.md's naming convention: subject IDs ending in 'A' are
9-month-olds, everyone else is 2-month-old.

Mask: age-specific fcgmask (nihpd_asym_02-05 for 2mo, nihpd_asym_08-11 for 9mo) -- fixed per run
according to that run's own age group (no per-run computation needed, since the whole batch for
a given --age-group run is single-age by construction).

Crop: a single fixed window per age group -- each mask's own bounding box, padded to a multiple
of 4. This is provably safe for every run within that age group without a per-run bounding-box
computation: since masked = raw * mask, masked's nonzero support is always a subset of the mask's
own support, so cropping to the mask's own (padded) bounding box can never clip real brain
content. Confirmed empirically for 2mo (see effective_bounding_box_analysis/run_bounding_boxes.csv:
the union of all 330 2mo videos-task runs' individual bounding boxes lands exactly on the 2mo
mask's own box). The two age groups' padded boxes are different absolute windows (9mo brains are
anatomically bigger) but happen to be the exact same final SIZE -- (60,72,56) for both -- so the
combined 2mo+9mo dataset is uniformly shaped without needing to reconcile the two.

Parallelized across a process pool (the earlier bbox-analysis job was fully sequential despite
requesting multiple CPUs); each run's crop offset is folded into its saved affine, so the output
stays correctly aligned in world space with everything else in this coordinate system (the
fcgmask, the Schaefer atlas, etc.) despite being a smaller array.
"""

import argparse
import csv
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

MASK_ROOT = Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/mask")

# Each age group's own mask and its own (already-computed) padded bounding box -- see module
# docstring. Both crop windows produce the same final (60,72,56) shape.
AGE_CONFIG = {
    "2mo": dict(
        mask_path=MASK_ROOT / "nihpd_asym_02-05_fcgmask_2mm.nii.gz",
        crop={"x": (17, 77), "y": (26, 98), "z": (11, 67)},
    ),
    "9mo": dict(
        mask_path=MASK_ROOT / "nihpd_asym_08-11_fcgmask_2mm.nii.gz",
        crop={"x": (18, 78), "y": (25, 97), "z": (11, 67)},
    ),
}

SUBJECT_DIR_RE = re.compile(r"_subject_id_(\w+)")
RUN_DIR_RE = re.compile(r"_run_(\d+)_session_(\d+)_task_name_(\w+)")

TASKS = ("videos", "rest10")

CSV_FIELDS = [
    "status", "subject_id", "session_id", "run_id", "task",
    "run_path", "out_path", "cropped_shape", "n_volumes", "elapsed_s", "message",
]


def discover_runs(runs_root, age_group):
    """Every videos/rest10 run under runs_root, for subjects matching age_group ('2mo': not
    ending in 'A'; '9mo': ending in 'A')."""
    runs = []
    for task in TASKS:
        pattern = f"_subject_id_*/_referencetype_standard/_run_*_session_*_task_name_{task}/*_bold_mcf_corrected_flirt.nii.gz"
        for p in runs_root.glob(pattern):
            subject_id = SUBJECT_DIR_RE.search(p.parts[-4]).group(1)
            is_9mo = subject_id.endswith("A")
            if (age_group == "9mo") == is_9mo:
                runs.append(p)
    return sorted(runs)


def cropped_affine(orig_affine, crop):
    """Shift the affine's translation to the crop window's own origin -- pure re-anchoring,
    no resampling, since this is an exact array slice."""
    x0, y0, z0 = crop["x"][0], crop["y"][0], crop["z"][0]
    new_affine = orig_affine.copy()
    new_affine[:, 3] = orig_affine @ np.array([x0, y0, z0, 1.0])
    return new_affine


def process_one_run(run_path_str, runs_root_str, out_root_str, age_group):
    """Load one run, apply that age group's fcgmask, crop to that age group's fixed window, save
    preserving the source's relative directory structure. Returns a result dict for CSV logging."""
    run_path = Path(run_path_str)
    runs_root = Path(runs_root_str)
    out_root = Path(out_root_str)
    cfg = AGE_CONFIG[age_group]
    crop = cfg["crop"]
    t0 = time.time()
    try:
        subject_id = SUBJECT_DIR_RE.search(run_path.parts[-4]).group(1)
        run_id, session_id, task = RUN_DIR_RE.search(run_path.parts[-2]).groups()

        mask_img = nib.load(cfg["mask_path"])
        mask = mask_img.get_fdata() > 0

        img = nib.load(run_path)
        data = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
        if data.shape[:3] != mask.shape:
            raise ValueError(f"run shape {data.shape[:3]} != mask shape {mask.shape}")

        masked = data * mask[..., None]
        cropped = masked[crop["x"][0]:crop["x"][1], crop["y"][0]:crop["y"][1], crop["z"][0]:crop["z"][1], :]

        new_affine = cropped_affine(img.affine, crop)
        out_header = img.header.copy()  # preserve TR and other source metadata (not derivable from the affine)
        out_header.set_data_shape(cropped.shape)
        out_header.set_data_dtype(np.float32)
        out_header.set_sform(new_affine, code=1)
        out_header.set_qform(new_affine, code=1)
        out_img = nib.Nifti1Image(cropped, new_affine, header=out_header)

        rel_path = run_path.relative_to(runs_root)
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, out_path)

        elapsed = time.time() - t0
        return dict(status="ok", subject_id=subject_id, session_id=session_id, run_id=run_id, task=task,
                    run_path=str(run_path), out_path=str(out_path), cropped_shape=str(cropped.shape),
                    n_volumes=data.shape[-1], elapsed_s=round(elapsed, 1), message="")
    except Exception as e:
        elapsed = time.time() - t0
        return dict(status="failed", subject_id="", session_id="", run_id="", task="",
                    run_path=str(run_path), out_path="", cropped_shape="", n_volumes="",
                    elapsed_s=round(elapsed, 1), message=str(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/normalized_to_common_space"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--age-group", choices=["2mo", "9mo"], default="2mo")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N runs (for testing)")
    args = parser.parse_args()

    runs = discover_runs(args.runs_root, args.age_group)
    if args.limit:
        runs = runs[: args.limit]
    print(f"found {len(runs)} {args.age_group} videos/rest10 runs under {args.runs_root}", flush=True)
    print(f"workers: {args.workers}, out_root: {args.out_root}, mask: {AGE_CONFIG[args.age_group]['mask_path'].name}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    t_start = time.time()
    with open(args.out_csv, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        futures = {pool.submit(process_one_run, str(r), str(args.runs_root), str(args.out_root), args.age_group): r for r in runs}
        for i, fut in enumerate(as_completed(futures)):
            row = fut.result()
            writer.writerow(row)
            f.flush()
            if row["status"] == "ok":
                n_ok += 1
                print(f"  [{i+1}/{len(runs)}] {row['subject_id']} ses-{row['session_id']} run-{row['run_id']} {row['task']}: "
                      f"{row['cropped_shape']} in {row['elapsed_s']}s", flush=True)
            else:
                n_failed += 1
                print(f"  [{i+1}/{len(runs)}] FAILED {row['run_path']}: {row['message']}", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\ndone: {n_ok} ok, {n_failed} failed, {total_elapsed/60:.1f} min total", flush=True)
    print(f"saved to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
