# maps_cyclegan_slurm — evaluation (200 epochs, checkpoint `latest`)

Generated on 2026-08-05 from SLURM training job 14849 and test job 14850, both run from
`pytorch-CycleGAN-and-pix2pix/`. This directory is standalone — it does not live inside
the training repo.

## Contents

- `loss_plots/` — training loss curves parsed from `checkpoints/maps_cyclegan_slurm/loss_log.txt`
  - `adversarial_losses.png` — D_A, D_B, G_A, G_B (smoothed)
  - `cycle_identity_losses.png` — cycle_A, cycle_B, idt_A, idt_B (smoothed)
  - `per_epoch_all_losses.png` — all 8 losses, per-epoch mean, epochs 1-200
- `test_results/` — full `test.py` output (cycle_gan model, both directions, 1098 test images,
  `--serial_batches --eval`, checkpoint = `latest` = epoch 200). Contains real/fake/reconstructed
  images for every test pair plus the auto-generated HTML viewer
  (`test_results/maps_cyclegan_slurm/test_latest/index.html`).
- `comparison_grids/` — 8-sample visual grids for quick side-by-side inspection
  - `comparison_A_to_B.png`: real_A (satellite) | fake_B (generated map) | real_B (GT map) | rec_A (cycle back)
  - `comparison_B_to_A.png`: real_B (map) | fake_A (generated satellite) | real_A (GT satellite) | rec_B (cycle back)
- `metrics/` — quantitative test-set evaluation
  - `per_image_metrics.csv` — PSNR/SSIM per test image, both directions
  - `metrics_summary.md` — aggregated means/stds (below)
- `plot_losses.py`, `compute_metrics.py`, `make_comparison_grid.py` — scripts used to produce the above (rerunnable on any other checkpoint/epoch)

## Quantitative results (full 1098-image test set)

| Direction | PSNR vs GT | SSIM vs GT | Cycle-recon PSNR | Cycle-recon SSIM |
|---|---|---|---|---|
| A→B (satellite → map) | 25.67 ± 4.24 | 0.708 ± 0.075 | 24.50 ± 1.78 | 0.827 ± 0.031 |
| B→A (map → satellite) | 15.05 ± 1.84 | 0.238 ± 0.081 | 35.07 ± 2.81 | 0.933 ± 0.021 |

**Reading these numbers:** satellite→map (A→B) is a well-posed, largely deterministic
problem (roads/water/buildings map to fixed colors), so pixel-wise PSNR/SSIM against
ground truth is meaningful and the scores are solid (SSIM ~0.71). Map→satellite (B→A) is
the harder, ill-posed direction — a single map tile is consistent with many plausible
photorealistic renderings (exact building color, shadow, vegetation), so low pixel-level
PSNR/SSIM there does **not** mean the model failed; see `comparison_B_to_A.png`, where the
generated satellite images are structurally coherent and follow the road layout, they just
don't pixel-align with the one specific ground-truth photo. The very high cycle-reconstruction
scores in both directions (SSIM 0.83 / 0.93) are the more trustworthy signal — they confirm
the two generators are consistent inverses of each other, which is the property CycleGAN is
actually optimized for.

## Training loss takeaway

Losses parsed from all 200 epochs (2192 logged steps): discriminator losses (D_A, D_B) stayed
in a healthy 0.03–0.35 band throughout (no collapse to 0, no runaway), while cycle-consistency
and identity losses dropped substantially in the first ~100 epochs and plateaued at low, stable
values through epoch 200 — consistent with a well-behaved, non-collapsed CycleGAN run. See
`loss_plots/per_epoch_all_losses.png` for the full picture.
