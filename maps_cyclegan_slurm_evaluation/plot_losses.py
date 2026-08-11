"""Parse pytorch-CycleGAN-and-pix2pix loss_log.txt and plot training curves."""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = sys.argv[1]
OUT_DIR = sys.argv[2]

LINE_RE = re.compile(
    r"epoch:\s*(\d+),\s*iters:\s*(\d+).*?"
    r"D_A:\s*([\d.]+).*?G_A:\s*([\d.]+).*?cycle_A:\s*([\d.]+).*?idt_A:\s*([\d.]+).*?"
    r"D_B:\s*([\d.]+).*?G_B:\s*([\d.]+).*?cycle_B:\s*([\d.]+).*?idt_B:\s*([\d.]+)"
)

records = []
with open(LOG_PATH) as f:
    for line in f:
        m = LINE_RE.search(line)
        if m:
            records.append([float(x) for x in m.groups()])

records = np.array(records)
epoch, iters = records[:, 0], records[:, 1]
D_A, G_A, cycle_A, idt_A, D_B, G_B, cycle_B, idt_B = records[:, 2:].T
step = np.arange(len(records))


def smooth(y, window=25):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_group(names, arrays, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, arr in zip(names, arrays):
        sm = smooth(arr)
        x = np.arange(len(sm))
        ax.plot(x, sm, label=name, linewidth=1.3)
    ax.set_xlabel("Logged step (x100 iters, smoothed window=25)")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{fname}", dpi=150)
    plt.close(fig)


plot_group(
    ["D_A", "D_B", "G_A", "G_B"],
    [D_A, D_B, G_A, G_B],
    "Adversarial losses (discriminators vs generators)",
    "adversarial_losses.png",
)

plot_group(
    ["cycle_A", "cycle_B", "idt_A", "idt_B"],
    [cycle_A, cycle_B, idt_A, idt_B],
    "Cycle-consistency and identity losses",
    "cycle_identity_losses.png",
)

# Per-epoch mean summary, one figure per loss, all 8 in a grid
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
arrays = [D_A, G_A, cycle_A, idt_A, D_B, G_B, cycle_B, idt_B]
max_epoch = int(epoch.max())
for ax, name, arr in zip(axes.flat, names, arrays):
    epoch_means = [arr[epoch == e].mean() for e in range(1, max_epoch + 1) if np.any(epoch == e)]
    ax.plot(range(1, len(epoch_means) + 1), epoch_means, linewidth=1.3)
    ax.set_title(name)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
fig.suptitle("Per-epoch mean losses — maps_cyclegan_slurm (200 epochs)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/per_epoch_all_losses.png", dpi=150)
plt.close(fig)

print(f"Parsed {len(records)} log lines spanning epochs 1-{max_epoch}")
print(f"Saved plots to {OUT_DIR}")
