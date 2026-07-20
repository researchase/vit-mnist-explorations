"""Show what mean- vs median-pooling does to the image -- explains the accuracy gap."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets

from tiny_pool import pool, MEAN, STD, CLASSES

ds = datasets.MNIST("/mnt/nvme1tb/data/mnist", train=False, download=True)
picks = []
for c in CLASSES:
    i = int((ds.targets == c).nonzero()[0])
    picks.append(i)

cols = [("original 28x28", None, None)]
for bs in (4, 7):
    for mode in ("mean", "median"):
        cols.append((f"{28//bs}x{28//bs}\n{mode}", bs, mode))

fig, axes = plt.subplots(len(picks), len(cols), figsize=(len(cols) * 1.9, len(picks) * 1.9))
for r, i in enumerate(picks):
    x = ds.data[i].float().div(255).sub(MEAN).div(STD).view(1, 1, 28, 28)
    for cc, (title, bs, mode) in enumerate(cols):
        img = x if bs is None else pool(x, bs, mode)
        ax = axes[r, cc]
        ax.imshow(img[0, 0].numpy(), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=10)
        if cc == 0:
            ax.set_ylabel(f"digit {int(ds.targets[i])}", fontsize=10)
fig.suptitle("Mean pooling keeps strokes; median pooling deletes them (strokes are the minority pixel)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("tiny_pool_viz.png", dpi=120)
print("wrote tiny_pool_viz.png")
