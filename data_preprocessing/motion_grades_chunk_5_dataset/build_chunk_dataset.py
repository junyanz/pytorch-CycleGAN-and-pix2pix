"""Build a motion-graded 5-volume chunk dataset from the brain-masked/cropped processed fMRI
data, using framewise displacement (FD) from motion_fwd/ to grade each chunk, then split into
train/val/test by subject.

Data sources:
  - FD:      motion_fwd/_subject_id_X/_run_NNN_session_N_task_name_TASK/_referencetype_standard/fd_power_2012.txt
  - Volumes: brain_masked_cropped_normalized_to_common_space/_subject_id_X/_referencetype_standard/
             _run_NNN_session_N_task_name_TASK/*.nii.gz  (already brain-masked, cropped to (60,72,56))
These two trees nest run/task directories in a DIFFERENT order relative to _referencetype_standard,
so runs are joined on (subject_id, session_id, run_id, task), not by literal path translation. A
handful of FD files have no matching processed volume (failed the source's '_corrected' processing
step, see data_preprocessing/brain_masked_cropped_2mo_dataset/build_dataset.py's investigation of
this same gap) -- skipped and logged, not treated as an error.

Scope: both tasks (videos, rest10) and both age groups (2mo, 9mo). Only 'rest10' exists in the
processed volume dataset (rest5 was never masked/cropped), so the two output task branches are
exactly 'videos' and 'rest10'.

Chunking: non-overlapping 5-volume windows [start, start+4], stepped by 5 per run independently.
FD-to-chunk alignment: FD[i] is the displacement of volume i+1 relative to volume i (so FD has
length T-1), giving each 5-volume chunk 4 internal FD values: FD[start:start+4].

Grading (locked in after iteration -- see conversation history for the full derivation):
  - Grade 1 (very low):  mean < 0.2mm AND max <= 0.25mm AND no FD spike in the safe-zone window
                          *before* the chunk (guards against motion-carryover/spin-history
                          contamination; before-only since the effect is forward-propagating).
                          Safe-zone spike threshold and window length are TASK-SPECIFIC (see
                          TASK_PARAMS below) -- videos and rest10 have very different baseline
                          motion levels and data abundance, so a uniform buffer either starves
                          videos or leaves rest10's Grade 1 pool too loosely screened.
  - Grade 2 (low):       0.2 <= mean < 0.5,  max <= 0.5
  - Grade 3 (moderate):  0.5 <= mean < 1.0,  max <= 1.0
  - Grade 4 (high):      1.0 <= mean < 2.0,  max <= 2.0
  - Grade 5 (very high): 2.0 <= mean < 4.0,  max <= 4.0
  - Grade 6 (extreme):   4.0 <  mean <= 10.0, max <= 10.0
  - Any chunk with an FD value > 10mm anywhere: excluded outright (catastrophic), never saved.
  - A chunk whose mean qualifies for a grade but whose max exceeds that grade's own cap is
    rejected entirely (not reassigned elsewhere).

Split: subject-wise 70/15/15 train/val/test, computed independently per task (the videos and
rest10 subject pools differ in membership and size), using SEED below. This seed was chosen by
searching 3000 seeds for the one that best balances each grade's train/val/test proportions
against the 70/15/15 target (weighted against leaving any split/grade cell empty) -- a plain
random split left some rare rest10 grades (3-6, which have very few contributing subjects) with
as few as 2 test chunks under some seeds.

Output: motion_grades_chunk_5_dataset/{train,val,test}/grade{1..6}/{videos,rest10}/ -- split
first, then grade, then task; flat within each leaf directory (no subject subfolders -- everything
encoded in the filename), plus a single metadata CSV logging every saved chunk with enough columns
for a dataloader to select/filter/join without touching the NIfTI files. Each run's processed
volume is loaded ONCE and all of its qualifying chunks are extracted in that same pass (not
reopened per chunk); parallelized across runs via a process pool.
"""

import argparse
import csv
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

FD_ROOT = Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/motion_fwd")
PROCESSED_ROOT = Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/brain_masked_cropped_normalized_to_common_space")
BUILD_LOG_DIR = Path("/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/brain_masked_cropped_2mo_dataset")

CHUNK_SIZE = 5
CATASTROPHIC_THRESHOLD = 10.0
GRADE1_MAX_CAP = 0.25
GRADE1_MEAN_CAP = 0.2

# Per-task Grade-1 safe-zone params: (spike threshold in mm, look-back window in volumes).
# Locked in after comparing several combinations -- see conversation history.
TASK_PARAMS = {
    "videos": dict(spike_threshold=0.5, buffer_volumes=15),
    "rest10": dict(spike_threshold=0.25, buffer_volumes=20),
}

TASKS = ("videos", "rest10")
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 2962

SUBJECT_DIR_RE = re.compile(r"_subject_id_(\w+)")
RUN_DIR_RE = re.compile(r"_run_(\d+)_session_(\d+)_task_name_(\w+)")

CSV_FIELDS = [
    "split", "grade", "task", "subject_id", "age_group", "session_id", "run_id",
    "chunk_start", "chunk_end", "chunk_size", "chunk_mean_fd", "chunk_max_fd",
    "n_volumes_in_run", "chunk_shape", "tr_seconds",
    "chunk_path", "chunk_relpath", "source_volume_path", "fd_path",
]


@dataclass
class Grade:
    name: str
    dir_name: str
    lo: float
    lo_inclusive: bool
    hi: float
    hi_inclusive: bool
    max_cap: float

    def chunk_qualifies(self, fd_chunk):
        mean_val = fd_chunk.mean()
        lo_ok = (mean_val >= self.lo) if self.lo_inclusive else (mean_val > self.lo)
        hi_ok = (mean_val <= self.hi) if self.hi_inclusive else (mean_val < self.hi)
        max_ok = fd_chunk.max() <= self.max_cap
        return lo_ok and hi_ok and max_ok


GRADES = [
    Grade("Grade 1", "grade1", 0.0, True, 0.2, False, GRADE1_MAX_CAP),
    Grade("Grade 2", "grade2", 0.2, True, 0.5, False, 0.5),
    Grade("Grade 3", "grade3", 0.5, True, 1.0, False, 1.0),
    Grade("Grade 4", "grade4", 1.0, True, 2.0, False, 2.0),
    Grade("Grade 5", "grade5", 2.0, True, 4.0, False, 4.0),
    Grade("Grade 6", "grade6", 4.0, False, 10.0, True, 10.0),
]


def load_fd(path):
    with open(path) as f:
        lines = f.readlines()[1:]
    return np.array([float(l.strip()) for l in lines if l.strip()])


def has_preceding_spike(fd, start, buffer_vols, threshold):
    lo_idx = max(0, start - buffer_vols)
    return bool((fd[lo_idx:start] >= threshold).any())


def classify_chunk(fd, fd_chunk, start, task):
    if fd_chunk.max() > CATASTROPHIC_THRESHOLD:
        return None, "catastrophic"
    params = TASK_PARAMS[task]
    for g in GRADES:
        if g.chunk_qualifies(fd_chunk):
            if g.name == "Grade 1" and has_preceding_spike(fd, start, params["buffer_volumes"], params["spike_threshold"]):
                return None, "grade1_buffer_rejected"
            return g, "ok"
    return None, "no_grade"  # mean qualified for a band but max exceeded that band's own cap


def find_processed_volume(subject_id, session_id, run_id, task):
    run_dir = PROCESSED_ROOT / f"_subject_id_{subject_id}" / "_referencetype_standard" / f"_run_{run_id}_session_{session_id}_task_name_{task}"
    if not run_dir.is_dir():
        return None
    matches = list(run_dir.glob("*.nii.gz"))
    return matches[0] if len(matches) == 1 else None


def unique_subjects_with_ok_task(task):
    """Every subject with at least one successfully built (masked/cropped) run for `task`, from
    the build_dataset.py logs (2mo + 9mo)."""
    subs = set()
    for csv_name in ("build_log.csv", "build_log_9mo.csv"):
        with open(BUILD_LOG_DIR / csv_name) as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok" and row["task"] == task:
                    subs.add(row["subject_id"])
    return sorted(subs)


def build_split_map():
    """subject -> split, computed independently per task (disjoint id namespaces would make a
    combined split meaningless here since a subject's videos and rest10 splits are allowed to
    differ -- see module docstring)."""
    split_map = {}
    for task in TASKS:
        subs = unique_subjects_with_ok_task(task)
        shuffled = subs[:]
        random.Random(SEED).shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * SPLIT_RATIOS[0])
        n_val = round(n * SPLIT_RATIOS[1])
        assignment = {}
        for sub in shuffled[:n_train]:
            assignment[sub] = "train"
        for sub in shuffled[n_train:n_train + n_val]:
            assignment[sub] = "val"
        for sub in shuffled[n_train + n_val:]:
            assignment[sub] = "test"
        split_map[task] = assignment
    return split_map


def process_one_run(fd_path_str, out_root_str, split_map):
    fd_path = Path(fd_path_str)
    out_root = Path(out_root_str)
    t0 = time.time()
    subject_id = SUBJECT_DIR_RE.search(fd_path.parts[-4]).group(1)
    run_id, session_id, task = RUN_DIR_RE.search(fd_path.parts[-3]).groups()
    age_group = "9mo" if subject_id.endswith("A") else "2mo"

    result = dict(subject_id=subject_id, session_id=session_id, run_id=run_id, task=task,
                   age_group=age_group, status="ok", message="", rows=[],
                   counts={(s, g.dir_name): 0 for s in SPLITS for g in GRADES},
                   catastrophic=0, grade1_buffer_rejected=0, no_grade=0)

    split = split_map[task].get(subject_id)
    if split is None:
        result["status"] = "skipped_no_split"
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    vol_path = find_processed_volume(subject_id, session_id, run_id, task)
    if vol_path is None:
        result["status"] = "skipped_no_volume"
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    try:
        fd = load_fd(fd_path)
        T = len(fd) + 1
        img = nib.load(vol_path)
        data = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
        if data.shape[-1] != T:
            raise ValueError(f"FD length T={T} does not match volume's T={data.shape[-1]}")
        zooms = img.header.get_zooms()
        tr_seconds = float(zooms[3]) if len(zooms) > 3 else None

        for start in range(0, T - CHUNK_SIZE + 1, CHUNK_SIZE):
            end = start + CHUNK_SIZE - 1
            fd_chunk = fd[start : start + CHUNK_SIZE - 1]
            if len(fd_chunk) < CHUNK_SIZE - 1:
                continue
            grade, status = classify_chunk(fd, fd_chunk, start, task)
            if status == "catastrophic":
                result["catastrophic"] += 1
                continue
            if status == "grade1_buffer_rejected":
                result["grade1_buffer_rejected"] += 1
                continue
            if status == "no_grade":
                result["no_grade"] += 1
                continue

            chunk_data = data[:, :, :, start : end + 1]  # (60,72,56,5)
            out_dir = out_root / split / grade.dir_name / task  # split, then grade, then task
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"sub-{subject_id}_ses-{session_id}_task-{task}_run-{run_id}_chunk-{start}-{end}.nii.gz"
            out_path = out_dir / fname
            chunk_relpath = str(Path(split) / grade.dir_name / task / fname)

            out_header = img.header.copy()
            out_header.set_data_shape(chunk_data.shape)
            out_header.set_data_dtype(np.float32)
            out_img = nib.Nifti1Image(chunk_data, img.affine, header=out_header)
            nib.save(out_img, out_path)

            result["counts"][(split, grade.dir_name)] += 1
            result["rows"].append(dict(
                split=split, grade=grade.name, task=task, subject_id=subject_id, age_group=age_group,
                session_id=session_id, run_id=run_id, chunk_start=start, chunk_end=end,
                chunk_size=CHUNK_SIZE,
                chunk_mean_fd=round(float(fd_chunk.mean()), 4), chunk_max_fd=round(float(fd_chunk.max()), 4),
                n_volumes_in_run=T, chunk_shape=str(tuple(chunk_data.shape)), tr_seconds=tr_seconds,
                chunk_path=str(out_path), chunk_relpath=chunk_relpath,
                source_volume_path=str(vol_path), fd_path=str(fd_path),
            ))
    except Exception as e:
        result["status"] = "failed"
        result["message"] = str(e)

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    split_map = build_split_map()
    for task in TASKS:
        counts = {"train": 0, "val": 0, "test": 0}
        for sub, sp in split_map[task].items():
            counts[sp] += 1
        print(f"{task} subject split (seed={SEED}): train={counts['train']} val={counts['val']} test={counts['test']}", flush=True)

    fd_files = []
    for task in TASKS:
        fd_files.extend(FD_ROOT.glob(f"_subject_id_*/_run_*_session_*_task_name_{task}/_referencetype_standard/fd_power_2012.txt"))
    fd_files = sorted(fd_files)
    if args.limit:
        fd_files = fd_files[: args.limit]
    print(f"found {len(fd_files)} FD files (videos+rest10, both age groups)", flush=True)
    print(f"workers: {args.workers}, out_root: {args.out_root}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    totals = {task: {(s, g.dir_name): 0 for s in SPLITS for g in GRADES} for task in TASKS}
    n_catastrophic = n_buffer_rejected = n_no_grade = 0
    n_skipped_no_volume = n_skipped_no_split = n_failed = 0
    t_start = time.time()

    with open(args.out_csv, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        futures = {pool.submit(process_one_run, str(fd), str(args.out_root), split_map): fd for fd in fd_files}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r["status"] == "skipped_no_split":
                n_skipped_no_split += 1
                print(f"  [{i+1}/{len(fd_files)}] SKIP (no split for subject) {r['subject_id']} {r['task']}", flush=True)
                continue
            if r["status"] == "skipped_no_volume":
                n_skipped_no_volume += 1
                print(f"  [{i+1}/{len(fd_files)}] SKIP (no processed volume) {r['subject_id']} ses-{r['session_id']} run-{r['run_id']} {r['task']}", flush=True)
                continue
            if r["status"] == "failed":
                n_failed += 1
                print(f"  [{i+1}/{len(fd_files)}] FAILED {r['subject_id']} ses-{r['session_id']} run-{r['run_id']} {r['task']}: {r['message']}", flush=True)
                continue
            for row in r["rows"]:
                writer.writerow(row)
            f.flush()
            for k, v in r["counts"].items():
                totals[r["task"]][k] += v
            n_catastrophic += r["catastrophic"]
            n_buffer_rejected += r["grade1_buffer_rejected"]
            n_no_grade += r["no_grade"]
            saved = sum(r["counts"].values())
            print(f"  [{i+1}/{len(fd_files)}] {r['subject_id']} ses-{r['session_id']} run-{r['run_id']} {r['task']} ({r['age_group']}): "
                  f"{saved} chunks saved in {r['elapsed_s']}s", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\ndone in {total_elapsed/60:.1f} min", flush=True)
    print(f"skipped (no split for subject): {n_skipped_no_split}", flush=True)
    print(f"skipped (no processed volume): {n_skipped_no_volume}", flush=True)
    print(f"failed: {n_failed}", flush=True)
    print(f"catastrophic (FD>10mm) excluded: {n_catastrophic}", flush=True)
    print(f"grade1 rejected by safe-zone buffer: {n_buffer_rejected}", flush=True)
    print(f"rejected: mean qualified, max exceeded that grade's cap: {n_no_grade}", flush=True)
    print(flush=True)
    for task in TASKS:
        print(f"=== {task}: chunks saved per split x grade ===", flush=True)
        for g in GRADES:
            row = [totals[task][(s, g.dir_name)] for s in SPLITS]
            print(f"  {g.name:10s}: train={row[0]:6d}  val={row[1]:6d}  test={row[2]:6d}  total={sum(row):6d}", flush=True)
        grand = [sum(totals[task][(s, g.dir_name)] for g in GRADES) for s in SPLITS]
        print(f"  {'TOTAL':10s}: train={grand[0]:6d}  val={grand[1]:6d}  test={grand[2]:6d}  total={sum(grand):6d}", flush=True)
        print(flush=True)
    grand_total = sum(sum(totals[t][(s, g.dir_name)] for s in SPLITS for g in GRADES) for t in TASKS)
    print(f"GRAND TOTAL across both tasks: {grand_total}", flush=True)
    print(f"\nsaved to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
