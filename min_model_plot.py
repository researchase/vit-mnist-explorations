import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (input, params, acc)
pts = [
    ("tiny 4x4", 159, 88.41), ("tiny 4x4", 179, 90.53), ("tiny 4x4", 411, 91.20), ("tiny 4x4", 483, 91.93),
    ("tiny 7x7", 291, 96.08), ("tiny 7x7", 311, 95.41), ("tiny 7x7", 675, 95.39), ("tiny 7x7", 747, 97.09),
    ("full 28x28", 351, 95.22), ("full 28x28", 371, 96.25), ("full 28x28", 795, 97.80), ("full 28x28", 867, 98.45),
]
colors = {"tiny 4x4": "#e6194B", "tiny 7x7": "#f2a900", "full 28x28": "#4363d8"}
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for name in colors:
    xs = [p for n, p, a in pts if n == name]
    ys = [a for n, p, a in pts if n == name]
    ax.scatter(xs, ys, s=70, color=colors[name], label=name, zorder=3, edgecolors="w")
ax.axhline(98, color="#2a2", ls="--", lw=1, label="98% target")
ax.annotate("smallest ≥98%\n867 params", xy=(867, 98.45), xytext=(1100, 96.5),
            arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.annotate("full-res dim=4 (371p, 96.3%)\nbeats tiny-4x4 dim=8 (483p, 91.9%)",
            xy=(371, 96.25), xytext=(150, 93.2), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#888"))
ax.set_xscale("log")
ax.set_xlabel("parameters (log)")
ax.set_ylabel("test accuracy (%)")
ax.set_title("Minimal-model frontier (1 block, MNIST 0/1/2)\nkeeping resolution + shrinking the model beats shrinking the image")
ax.legend(loc="lower right")
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig("min_model.png", dpi=120)
print("wrote min_model.png")
