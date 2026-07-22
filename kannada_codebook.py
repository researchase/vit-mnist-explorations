"""Build a discrete 'visual vocabulary' for Kannada digits {0,1,2}.

K-means over all 4x4 patches of the Kannada training images -> K centroid
patches (the alphabet). Sanity check: reconstruct held-out digits using only
codebook entries and save a side-by-side grid.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/mnt/nvme1tb/data/kannada_mnist/Kannada_MNIST_npz/Kannada_MNIST"
K = 64          # codebook size
P = 4           # patch size -> 7x7 = 49 patches per image
CLASSES = (0, 1, 2)
dev = "cuda"

def load(split):
    X = np.load(f"{DATA}/X_kannada_MNIST_{split}.npz")["arr_0"]
    y = np.load(f"{DATA}/y_kannada_MNIST_{split}.npz")["arr_0"]
    m = np.isin(y, CLASSES)
    return X[m].astype(np.float32) / 255.0, y[m]

def to_patches(X):
    # (N,28,28) -> (N,49,16), raster order
    t = torch.from_numpy(X).unfold(1, P, P).unfold(2, P, P)   # N,7,7,4,4
    return t.reshape(-1, 49, P * P)

def kmeans(x, k, iters=50, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=dev)[:k]]
    for _ in range(iters):
        d = torch.cdist(x, c)                 # N,k
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c, a

if __name__ == "__main__":
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    print(f"train {Xtr.shape} test {Xte.shape}  classes {np.bincount(ytr)}")

    ptr = to_patches(Xtr).to(dev)                       # N,49,16
    flat = ptr.reshape(-1, P * P)
    # subsample for k-means speed
    idx = torch.randperm(len(flat))[:200_000].to(dev)
    code, _ = kmeans(flat[idx], K)
    # sort codebook by brightness so token ids are interpretable
    code = code[code.sum(1).argsort()]
    torch.save(code.cpu(), "kannada_codebook.pt")

    # reconstruction sanity check on test digits
    pte = to_patches(Xte).to(dev)
    d = torch.cdist(pte.reshape(-1, P * P), code)
    tok = d.argmin(1).reshape(-1, 49)                   # token ids per image
    rec = code[tok].reshape(-1, 7, 7, P, P).permute(0, 1, 3, 2, 4).reshape(-1, 28, 28)
    mse = ((rec.cpu().numpy() - Xte) ** 2).mean()
    used = len(tok.unique())
    print(f"codebook K={K}: recon MSE {mse:.5f}, codes used on test: {used}/{K}")

    # rows 0,2,4 originals / 1,3,5 recons, 12 examples per class
    fig, axes = plt.subplots(6, 12, figsize=(12, 6.5))
    for r, c in enumerate(CLASSES):
        ids = np.where(yte == c)[0][:12]
        for j, i in enumerate(ids):
            axes[2 * r][j].imshow(Xte[i], cmap="gray")
            axes[2 * r + 1][j].imshow(rec[i].cpu(), cmap="gray")
            axes[2 * r][j].axis("off"); axes[2 * r + 1][j].axis("off")
        axes[2 * r][0].set_title(f"kannada {c}: original (top) vs {K}-token recon (bottom)",
                                 loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig("kannada_codebook_recon.png", dpi=110)

    # the alphabet itself
    fig, axes = plt.subplots(4, 16, figsize=(12, 3.2))
    for j in range(K):
        ax = axes[j // 16][j % 16]
        ax.imshow(code[j].reshape(P, P).cpu(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle("the Kannada visual alphabet: 64 patch tokens (sorted by ink)")
    plt.tight_layout()
    plt.savefig("kannada_alphabet.png", dpi=110)
    print("saved kannada_codebook.pt, kannada_codebook_recon.png, kannada_alphabet.png")
