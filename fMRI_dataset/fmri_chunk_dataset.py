"""Unpaired fMRI motion-correction dataset for CycleGAN training.

Domain A = Grade 1 (very low motion -- the "clean" reference). Domain B = Grades 2-6 pooled
(motion-corrupted, any severity from mild to extreme -- deliberately not split by severity, so the
model sees the full diversity of real motion artifacts as one domain). Unpaired: A and B chunks are
never assumed to correspond to the same subject/run/timepoint (see models3D/cycle_gan_model.py,
which trains on exactly this A/B convention already).

Loads 5-volume chunks built by data_preprocessing/motion_grades_chunk_5_dataset/build_chunk_dataset.py
(see chunk_metadata.csv for the full row schema) and normalizes each chunk using its SOURCE RUN's
precomputed robust_p5p95 stats (median, scale) from run_normalization_stats.csv -- see
compute_run_normalization_stats.py, which must be run once before this dataset is usable (stats are
computed per run, not per chunk, since 5 timepoints is too few for a stable percentile estimate).
Normalization is unclipped (see ../data_preprocessing/normalization_study.ipynb's "Chosen strategy"
section): the project's generator uses use_residual=True, so the output is never Tanh-bounded, which
removes the usual reason to clip -- unclipped keeps normalization exactly invertible.

Same directory layout for every split ({split}/grade{1-6}/{videos,rest10}/*.nii.gz), so one dataset
class handles train/val/test -- just pass a different `split`.

Returned tensors are shaped (C=1, T=5, H, W, D) = (1, 5, 60, 72, 56), matching the
(B, C, T, H, W, D) convention required by ResnetGenerator's temporal module
(models3D/networks.py, use_temporal_module=True) once the DataLoader adds the batch dimension.
H, W, D correspond to the NIfTI (X, Y, Z) axes respectively, verified via nib.aff2axcodes as
('R', 'A', 'S') -- so H is the left-right axis (safe to mirror-flip as augmentation), W is
anterior-posterior and D is superior-inferior (NOT safe to flip -- not anatomically valid).
"""
import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

GRADE_A = "Grade 1"
GRADES_B = ("Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6")
SPLITS = ("train", "val", "test")

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNK_METADATA_CSV = THIS_DIR.parent / "data_preprocessing/motion_grades_chunk_5_dataset/chunk_metadata.csv"
DEFAULT_RUN_STATS_CSV = THIS_DIR / "run_normalization_stats.csv"


def _load_run_stats(run_stats_csv):
    """(subject_id, session_id, run_id, task) -> (median, scale), from compute_run_normalization_stats.py's output."""
    stats = {}
    with open(run_stats_csv) as f:
        for row in csv.DictReader(f):
            key = (row["subject_id"], row["session_id"], row["run_id"], row["task"])
            stats[key] = (float(row["median"]), float(row["scale"]))
    return stats


def _load_chunk_rows(chunk_metadata_csv, split, task=None):
    """All chunk_metadata.csv rows for this split (optionally restricted to one task), grouped by grade."""
    rows_by_grade = {}
    with open(chunk_metadata_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if task is not None and row["task"] != task:
                continue
            rows_by_grade.setdefault(row["grade"], []).append(row)
    return rows_by_grade


def _run_key(row):
    return (row["subject_id"], row["session_id"], row["run_id"], row["task"])


class FMRIUnpairedGradeDataset(Dataset):
    """Unpaired (Grade 1) <-> (Grade 2-6 pooled) dataset for one split.

    Grade 1 (A) and Grade 2-6-pooled (B) are imbalanced (e.g. train: A=9350, B=13033). Rather than
    padding the smaller domain up to the larger one (which would repeat some A chunks within a
    single epoch) or sampling B with replacement on every call (which can repeat some B chunks and
    skip others within an epoch, with no coverage guarantee), this dataset:
      - sets epoch length to min(A_size, B_size) -- so a full pass through the dataset shows every
        sample of the SMALLER domain exactly once, no repeats, no padding
      - for the LARGER domain, uses a fresh min(A_size,B_size)-length random permutation (no
        repeats within that epoch) each epoch, via set_epoch() -- see its docstring for why this
        needs to be called explicitly rather than done automatically.

    Set full_coverage=True (recommended for val/test) to instead deterministically cycle through
    ALL of both domains every pass, with no randomness at all -- see full_coverage's docstring for
    why this is preferable to either a fixed subsample or a randomly-varying one for evaluation.

    Returns a dict matching the convention models3D/cycle_gan_model.py's set_input() already
    expects ({"A", "B", "A_paths", "B_paths"}), plus "A_meta"/"B_meta" dicts with subject/session/
    run/task/grade/FD info and the (median, scale) used to normalize that sample -- needed to
    denormalize model output back to raw intensity units later (see denormalize_values below).
    """

    def __init__(self, split, chunk_metadata_csv=DEFAULT_CHUNK_METADATA_CSV,
                 run_stats_csv=DEFAULT_RUN_STATS_CSV, task=None, flip_prob=0.0, base_seed=0,
                 full_coverage=False):
        """
        split          -- 'train' | 'val' | 'test'
        task           -- None (both videos+rest10 pooled) or 'videos'/'rest10' to restrict to one
        flip_prob      -- probability of a left-right mirror flip (the only anatomically-safe spatial
                           augmentation here -- see module docstring). 0.0 = off (default). Applied
                           independently to A and B, each drawn its own coin flip.
        base_seed      -- combined with the epoch number in set_epoch() to seed that epoch's random
                           permutation of the larger domain (ignored if full_coverage=True). Fixed
                           default (not random) so results are reproducible given the same
                           (base_seed, epoch).
        full_coverage  -- if True: __len__ = max(A_size, B_size), and BOTH domains are indexed by
                           plain `index % size` (deterministic cycling, no randomness, set_epoch()
                           becomes a no-op). Every sample from both domains is used in every single
                           pass, in the exact same fixed order every time. Recommended for val/test:
                           none of this project's CycleGAN losses (cycle-consistency, identity, GAN)
                           depend on which A got paired with which B -- each is computed per-sample
                           within its own domain -- so there was never a real need to fix the PAIRING
                           for reproducible metrics, only to use the same COMPLETE set of samples
                           every time. full_coverage gives you that outright: 100% of the data, zero
                           sampling variance, every evaluation call -- strictly better than either a
                           fixed narrow subsample (misses most of the larger domain) or a randomly
                           varying one (adds evaluation noise) for the default (train) mode.

        IMPORTANT (full_coverage=False only): the smaller domain relies on the DataLoader's own
        `shuffle=True` for its per-epoch order (this class hands it out in dataset order,
        index-for-index) -- pass shuffle=True when wrapping this in a DataLoader, or the smaller
        domain never gets reshuffled.
        """
        assert split in SPLITS, f"split must be one of {SPLITS}, got {split!r}"
        assert 0.0 <= flip_prob <= 1.0
        self.split = split
        self.flip_prob = flip_prob
        self.base_seed = base_seed
        self.full_coverage = full_coverage

        self.run_stats = _load_run_stats(run_stats_csv)

        rows_by_grade = _load_chunk_rows(chunk_metadata_csv, split, task=task)
        self.A_rows = rows_by_grade.get(GRADE_A, [])
        self.B_rows = [row for g in GRADES_B for row in rows_by_grade.get(g, [])]
        self.A_size = len(self.A_rows)
        self.B_size = len(self.B_rows)
        if self.A_size == 0 or self.B_size == 0:
            raise ValueError(f"split={split!r} task={task!r}: A_size={self.A_size}, B_size={self.B_size} -- both must be > 0")

        missing = {_run_key(r) for r in self.A_rows + self.B_rows if _run_key(r) not in self.run_stats}
        if missing:
            raise ValueError(
                f"{len(missing)} run(s) referenced by chunks in split={split!r} have no entry in "
                f"{run_stats_csv} -- re-run compute_run_normalization_stats.py. Example missing key: {next(iter(missing))}"
            )

        # Whichever domain is bigger gets index-remapped per epoch; the smaller domain is indexed
        # directly (0..len(self)-1) and relies on the DataLoader's shuffle=True for variety.
        # (Irrelevant when full_coverage=True, which uses plain modulo cycling on both sides instead.)
        self._b_is_larger = self.B_size >= self.A_size
        self._epoch_len = max(self.A_size, self.B_size) if full_coverage else min(self.A_size, self.B_size)
        self._larger_size = self.B_size if self._b_is_larger else self.A_size
        self._epoch = None
        self.set_epoch(0)

    def __len__(self):
        return self._epoch_len

    def set_epoch(self, epoch):
        """Call once per epoch, in the MAIN process, before creating that epoch's DataLoader
        iterator (i.e. before the `for batch in loader:` line) -- NOT inside the training loop body
        after iteration has already started. This matters because num_workers>0 forks a separate
        copy of this Dataset into each worker process at iterator-creation time; if the per-epoch
        permutation were instead computed lazily/randomly inside __getitem__, each worker would
        pick its own independent permutation and the "unique within this epoch" guarantee would
        break across workers. Calling set_epoch() first means every freshly-forked worker inherits
        the exact same deterministic (epoch-seeded) permutation. Same pattern and same reason as
        torch.utils.data.distributed.DistributedSampler.set_epoch().

        Draws a fresh min(A_size,B_size)-length permutation of the larger domain's indices for this
        epoch (independent of other epochs' permutations, not a persisting non-repeating queue --
        simpler and trivially resumable after a checkpoint restart). No repeats within an epoch;
        different subset/order than the previous epoch with overwhelming probability.

        No-op when full_coverage=True -- there's no per-epoch randomness to seed in that mode.
        """
        self._epoch = epoch
        if self.full_coverage:
            return
        g = torch.Generator().manual_seed(self.base_seed + epoch)
        self._larger_epoch_indices = torch.randperm(self._larger_size, generator=g)[: self._epoch_len].tolist()

    def _load_chunk_tensor(self, row):
        """Load one chunk NIfTI, normalize with its source run's (median, scale), optionally flip.
        Returns a (1, T, H, W, D) float32 tensor."""
        median, scale = self.run_stats[_run_key(row)]

        img = nib.load(row["chunk_path"])
        data = np.asarray(img.dataobj, dtype=np.float32)  # (H, W, D, T) = (60, 72, 56, 5)
        mask = data[..., 0] != 0

        normalized = np.zeros_like(data, dtype=np.float32)
        normalized[mask] = (data[mask] - median) / scale

        tensor = torch.from_numpy(normalized).permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, H, W, D)

        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            tensor = torch.flip(tensor, dims=[2])  # dim 2 = H = left-right axis (see module docstring)

        return tensor, median, scale

    def _meta(self, row, median, scale):
        return dict(
            subject_id=row["subject_id"], session_id=row["session_id"], run_id=row["run_id"],
            task=row["task"], grade=row["grade"], chunk_start=int(row["chunk_start"]),
            chunk_end=int(row["chunk_end"]), chunk_mean_fd=float(row["chunk_mean_fd"]),
            chunk_max_fd=float(row["chunk_max_fd"]), source_volume_path=row["source_volume_path"],
            median=median, scale=scale,
        )

    def __getitem__(self, index):
        if self.full_coverage:
            A_row = self.A_rows[index % self.A_size]
            B_row = self.B_rows[index % self.B_size]
        elif self._b_is_larger:
            A_row = self.A_rows[index]
            B_row = self.B_rows[self._larger_epoch_indices[index]]
        else:
            B_row = self.B_rows[index]
            A_row = self.A_rows[self._larger_epoch_indices[index]]

        A_tensor, A_median, A_scale = self._load_chunk_tensor(A_row)
        B_tensor, B_median, B_scale = self._load_chunk_tensor(B_row)

        return {
            "A": A_tensor,
            "B": B_tensor,
            "A_paths": A_row["chunk_path"],
            "B_paths": B_row["chunk_path"],
            "A_meta": self._meta(A_row, A_median, A_scale),
            "B_meta": self._meta(B_row, B_median, B_scale),
        }


def denormalize_values(values, median, scale):
    """Inverse of the normalization applied in _load_chunk_tensor -- exact, since robust_p5p95 is
    unclipped. `values` should be brain-voxel values only (see denormalize_tensor for the
    whole-tensor convenience wrapper). Matches normalization_study.ipynb's denormalize_robust_p5p95."""
    return values * scale + median


def denormalize_tensor(tensor, mask, median, scale):
    """Convenience wrapper for a full (C,T,H,W,D) (or batched) tensor: applies denormalize_values
    only where `mask` is True, leaving background at 0. `mask` must be the SAME brain mask used at
    normalization time (or reconstructed via chunk_data[...,0] != 0 from the corresponding raw
    chunk) -- applying the inverse to the WHOLE tensor unconditionally would incorrectly turn
    background's 0 into `median` (this exact bug was hit and fixed earlier in this project for
    robust_normalize() in dataset_creation.ipynb)."""
    out = torch.zeros_like(tensor)
    out[mask] = denormalize_values(tensor[mask], median, scale)
    return out


def worker_init_fn(worker_id):
    """Pass to DataLoader(..., worker_init_fn=worker_init_fn) when using num_workers > 0.
    Without this, every worker process inherits the SAME numpy/random RNG state from the fork,
    so B's random index and the flip augmentation would be identically "random" across workers --
    a well-known, easy-to-miss PyTorch DataLoader pitfall."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
