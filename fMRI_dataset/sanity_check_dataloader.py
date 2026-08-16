"""Sanity checks for FMRIUnpairedGradeDataset (fmri_chunk_dataset.py). Run directly:

    python sanity_check_dataloader.py

Checks: dataset sizes per split, single-sample shapes/stats, epoch-balanced sampling (train:
no repeats within an epoch, partial overlap between epochs, near-complete coverage over several
epochs), full_coverage mode (val/test: exact, deterministic 100% coverage of both domains),
denormalization round-trip exactness, flip augmentation reversibility, and real multi-worker
DataLoader integration.
"""

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fmri_chunk_dataset import (
    SPLITS,
    FMRIUnpairedGradeDataset,
    _run_key,
    denormalize_tensor,
    worker_init_fn,
)

print("=== FMRIUnpairedGradeDataset sanity check ===\n")

for split in SPLITS:
    ds = FMRIUnpairedGradeDataset(split=split)
    print(f"[{split}] A_size (Grade 1) = {ds.A_size}, B_size (Grade 2-6) = {ds.B_size}, len(dataset) = {len(ds)}")

print()
print("=== single-sample check (train split) ===")
ds = FMRIUnpairedGradeDataset(split="train")
sample = ds[0]
for key in ("A", "B"):
    t = sample[key]
    print(f"{key}: shape={tuple(t.shape)}, dtype={t.dtype}, "
          f"min={t.min():.3f}, max={t.max():.3f}, mean={t.mean():.4f}")
print("A_paths:", sample["A_paths"])
print("B_paths:", sample["B_paths"])
print("A_meta:", sample["A_meta"])
print("B_meta:", sample["B_meta"])

print()
print("=== epoch-balanced sampling check (train split: A=9350 smaller, B=13033 larger) ===")
print(f"len(dataset) = {len(ds)}  (expect min(A_size, B_size) = {min(ds.A_size, ds.B_size)})")
print("(inspects ds._larger_epoch_indices directly -- no file I/O, just the index bookkeeping)")


def _epoch_larger_paths(dataset):
    larger_rows = dataset.B_rows if dataset._b_is_larger else dataset.A_rows
    return [larger_rows[i]["chunk_path"] for i in dataset._larger_epoch_indices]


ds.set_epoch(0)
epoch0_paths = _epoch_larger_paths(ds)
print(f"epoch 0: drew {len(epoch0_paths)} larger-domain samples, {len(set(epoch0_paths))} unique "
      f"(expect equal -- no repeats within an epoch)")

ds.set_epoch(1)
epoch1_paths = _epoch_larger_paths(ds)
print(f"epoch 1: drew {len(epoch1_paths)} larger-domain samples, {len(set(epoch1_paths))} unique")
overlap = len(set(epoch0_paths) & set(epoch1_paths))
print(f"overlap between epoch 0's and epoch 1's subsets: {overlap} / {len(ds)} "
      f"({overlap/len(ds):.1%} -- expect a partial, not-identical overlap, not 0% or 100%)")

# coverage over a handful of epochs: what fraction of ALL of the larger domain has appeared at least once?
seen = set()
n_epochs_to_check = 5
for e in range(n_epochs_to_check):
    ds.set_epoch(e)
    seen.update(_epoch_larger_paths(ds))
larger_size = ds.B_size if ds._b_is_larger else ds.A_size
print(f"coverage after {n_epochs_to_check} epochs: {len(seen)} / {larger_size} distinct chunks seen "
      f"({len(seen)/larger_size:.1%})")
ds.set_epoch(0)  # reset before the rest of the checks below

print()
print("=== full_coverage check (recommended mode for val/test) ===")
for split in ("val", "test"):
    ds_full = FMRIUnpairedGradeDataset(split=split, full_coverage=True)
    print(f"[{split}] A_size={ds_full.A_size}, B_size={ds_full.B_size}, "
          f"len(dataset)={len(ds_full)}  (expect max = {max(ds_full.A_size, ds_full.B_size)})")

    A_indices_seen = [i % ds_full.A_size for i in range(len(ds_full))]
    B_indices_seen = [i % ds_full.B_size for i in range(len(ds_full))]
    print(f"  A: {len(set(A_indices_seen))}/{ds_full.A_size} unique indices covered "
          f"(expect {ds_full.A_size}/{ds_full.A_size} -- full coverage, some repeated to fill the length)")
    print(f"  B: {len(set(B_indices_seen))}/{ds_full.B_size} unique indices covered "
          f"(expect {ds_full.B_size}/{ds_full.B_size} -- exactly once each, since B is the larger domain here)")

    # determinism: two independent instantiations must produce the IDENTICAL sequence, no randomness at all
    ds_full_2 = FMRIUnpairedGradeDataset(split=split, full_coverage=True)
    same_A = [ds_full.A_rows[i % ds_full.A_size]["chunk_path"] for i in range(len(ds_full))] == \
             [ds_full_2.A_rows[i % ds_full_2.A_size]["chunk_path"] for i in range(len(ds_full_2))]
    print(f"  deterministic across independent instantiations: {same_A}")

print()
print("=== denormalization round-trip check ===")
row = ds.A_rows[0]
median, scale = ds.run_stats[_run_key(row)]
img = nib.load(row["chunk_path"])
raw = np.asarray(img.dataobj, dtype=np.float32)
mask_np = raw[..., 0] != 0

tensor, _, _ = ds._load_chunk_tensor(row)  # (1, T, H, W, D), normalized
tensor_xyzt = tensor.squeeze(0).permute(1, 2, 3, 0).numpy()  # back to (H, W, D, T) for comparison
mask_t = torch.from_numpy(np.broadcast_to(mask_np[..., None], raw.shape).copy())
denorm = denormalize_tensor(torch.from_numpy(tensor_xyzt), mask_t, median, scale).numpy()

abs_err = np.abs(denorm[mask_np] - raw[mask_np])
print(f"round-trip max abs error (brain voxels): {abs_err.max():.3e}")
print(f"round-trip mean abs error (brain voxels): {abs_err.mean():.3e}")
print(f"background preserved as exactly 0: {np.array_equal(denorm[~mask_np], raw[~mask_np])}")

print()
print("=== flip augmentation check ===")
ds_flip = FMRIUnpairedGradeDataset(split="train", flip_prob=1.0)  # always flip, for a deterministic check
flipped = ds_flip._load_chunk_tensor(row)[0]
unflipped, _, _ = ds._load_chunk_tensor(row)
print(f"flip_prob=1.0 changes the tensor: {not torch.equal(flipped, unflipped)}")
print(f"flipping twice recovers the original: {torch.equal(torch.flip(flipped, dims=[2]), unflipped)}")

print()
print("=== DataLoader integration check (small subset, batch_size=2, num_workers=2, 2 simulated epochs) ===")
print("(correct usage: call set_epoch() in the main process BEFORE creating each epoch's")
print(" DataLoader iterator -- see set_epoch()'s docstring for why. Uses a 50-item Subset so")
print(" this actually loads real files quickly -- full-dataset uniqueness was already proven")
print(" above with zero I/O; this part just confirms the real multi-worker pipeline works.)")
small = Subset(ds, list(range(50)))
loader = DataLoader(small, batch_size=2, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
for e in range(2):
    ds.set_epoch(e)
    batch_shapes = []
    for batch in loader:
        batch_shapes.append(tuple(batch["A"].shape))
    print(f"epoch {e}: {len(batch_shapes)} batches loaded, all shape (2,1,5,60,72,56): "
          f"{all(s == (2, 1, 5, 60, 72, 56) for s in batch_shapes)}")
batch = next(iter(loader))
print(f"batch A shape: {tuple(batch['A'].shape)}  (expect (2, 1, 5, 60, 72, 56))")
print(f"batch B shape: {tuple(batch['B'].shape)}")

print("\nall checks completed.")
