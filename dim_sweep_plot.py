import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dim   = [4, 8, 16, 32, 64, 128]
params= [371, 867, 2243, 6531, 21251, 75267]
acc   = [96.44, 98.72, 98.86, 99.00, 98.82, 98.72]
sd    = [0.23, 0.01, 0.07, 0.18, 0.23, 0.28]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(range(len(dim)), acc, yerr=sd, fmt="o-", color="#4363d8", lw=2, label="accuracy")
ax.set_xticks(range(len(dim)))
ax.set_xticklabels([f"dim={d}\n{p:,}p" for d, p in zip(dim, params)], fontsize=9)
ax.set_ylabel("test accuracy (%)", color="#4363d8")
ax.set_ylim(95.5, 99.6)
ax.axhline(acc[-2], color="#aaa", ls=":", lw=1)
ax.annotate("dim=8 matches dim=64\nat 1/24th the params",
            xy=(1, acc[1]), xytext=(2.2, 96.6),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10)
ax2 = ax.twinx()
ax2.plot(range(len(dim)), params, "s--", color="#e6194B", alpha=.6, label="params")
ax2.set_ylabel("parameters (log)", color="#e6194B")
ax2.set_yscale("log")
ax.set_title("Embedding dim sweep (1 block, 4 heads, MNIST 0/1/2)\naccuracy plateaus by dim=8; params grow ~quadratically")
fig.tight_layout()
fig.savefig("dim_sweep.png", dpi=120)
print("wrote dim_sweep.png")
