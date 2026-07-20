"""
Find the smallest attention-only ViT (1 block) that still clears ~98% on MNIST {0,1,2}.
Grid: input {full 28x28 learned / tiny 7x7 mean-pool / tiny 4x4 mean-pool}
      x dim {4,8} x heads {1,2}.  Ranked by parameter count.
"""
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

import viz_journey as vj
from train_gpu import ViT
from viz_journey_attn_only import AttnOnlyTransformer
from tiny_pool import load_tensors, pool

DEVICE = vj.DEVICE


def build(image_size, patch_size, dim, heads):
    dh = dim // heads
    m = ViT(image_size=image_size, patch_size=patch_size, num_classes=3, channels=1,
            dim=dim, depth=1, heads=heads, dim_head=dh, mlp_dim=dim * 2).to(DEVICE)
    m.transformer = AttnOnlyTransformer(dim=dim, depth=1, heads=heads, dim_head=dh).to(DEVICE)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(Xtr, ytr, Xte, yte, image_size, patch_size, dim, heads, epochs=6, seeds=(0, 1, 2), bs=128):
    accs, params = [], None
    N = Xtr.shape[0]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = build(image_size, patch_size, dim, heads); params = n_params(m)
        opt = optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(N, device=DEVICE)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = F.nll_loss(F.log_softmax(m(Xtr[idx]), dim=1), ytr[idx])
                loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            accs.append((m(Xte).argmax(1) == yte).float().mean().item() * 100)
    return params, np.mean(accs), np.std(accs)


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    ytr, yte = ytr.to(DEVICE), yte.to(DEVICE)
    inputs = [
        ("full 28x28", Xtr.to(DEVICE), Xte.to(DEVICE), 28, 7),
        ("tiny 7x7", pool(Xtr, 4, "mean").to(DEVICE), pool(Xte, 4, "mean").to(DEVICE), 7, 1),
        ("tiny 4x4", pool(Xtr, 7, "mean").to(DEVICE), pool(Xte, 7, "mean").to(DEVICE), 4, 1),
    ]
    rows = []
    for name, Xa, Xb, isz, ps in inputs:
        for dim in (4, 8):
            for heads in (1, 2):
                p, mu, sd = run(Xa, ytr, Xb, yte, isz, ps, dim, heads)
                rows.append((name, dim, heads, p, mu, sd))
    rows.sort(key=lambda r: r[3])
    print(f"\n{'input':11s} {'dim':>4s} {'heads':>5s} {'params':>7s} {'accuracy':>16s}  {'>=98%':>5s}")
    print("-" * 56)
    for name, dim, heads, p, mu, sd in rows:
        flag = "  ✓" if mu >= 98 else ""
        print(f"{name:11s} {dim:>4d} {heads:>5d} {p:>7,d}   {mu:>6.2f}% ± {sd:.2f}{flag}")
    best = min((r for r in rows if r[4] >= 98), key=lambda r: r[3], default=None)
    if best:
        print(f"\nsmallest >=98%: {best[0]}, dim={best[1]}, heads={best[2]}  "
              f"-> {best[3]:,} params @ {best[4]:.2f}%")


if __name__ == "__main__":
    main()
