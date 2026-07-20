"""
Twist: pool each patch to a SINGLE pixel (mean vs median) BEFORE the transformer, shrinking
the image and replacing the learned patch-embedding Linear(patch_dim->dim) with Linear(1->dim).

Model is fixed at the winning config: attention-only, 1 block, 4 heads (dim=64, dim_head=16).

Configs compared on MNIST {0,1,2}:
  baseline : 28x28, learned patch_size=7 embedding  (16 tokens, patch_dim=49)
  tiny 4x4 : pool each 7x7 patch -> 1 px  -> 4x4 image, patch_size=1 (16 tokens, patch_dim=1)
  tiny 7x7 : pool each 4x4 patch -> 1 px  -> 7x7 image, patch_size=1 (49 tokens, patch_dim=1)
each tiny config is run with mean-pool and median-pool.
"""
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import datasets

import viz_journey as vj
from train_gpu import ViT
from viz_journey_attn_only import AttnOnlyTransformer

DEVICE = vj.DEVICE
CLASSES = vj.CLASSES
MEAN, STD = 0.1307, 0.3081


def load_tensors():
    tr = datasets.MNIST("/mnt/nvme1tb/data/mnist", train=True, download=True)
    te = datasets.MNIST("/mnt/nvme1tb/data/mnist", train=False, download=True)

    def prep(ds):
        mask = torch.isin(ds.targets, torch.tensor(CLASSES))
        X = ds.data[mask].float().div(255).sub(MEAN).div(STD).unsqueeze(1)  # [N,1,28,28]
        y = ds.targets[mask].clone()
        return X, y

    return prep(tr), prep(te)


def pool(X, bs, mode):
    """Pool non-overlapping bs x bs blocks to one value -> [N,1,28/bs,28/bs]."""
    N, C, H, W = X.shape
    blocks = X.unfold(2, bs, bs).unfold(3, bs, bs)          # [N,C,H/bs,W/bs,bs,bs]
    blocks = blocks.reshape(N, C, H // bs, W // bs, bs * bs)
    if mode == "mean":
        return blocks.mean(-1)
    return torch.quantile(blocks, 0.5, dim=-1)              # true (interpolated) median


def build(image_size, patch_size, heads=4):
    m = ViT(image_size=image_size, patch_size=patch_size, num_classes=len(CLASSES),
            channels=1, dim=64, depth=1, heads=heads, dim_head=16, mlp_dim=128).to(DEVICE)
    m.transformer = AttnOnlyTransformer(dim=64, depth=1, heads=heads, dim_head=16).to(DEVICE)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(Xtr, ytr, Xte, yte, image_size, patch_size, epochs=6, seeds=(0, 1, 2), bs_train=128):
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xte, yte = Xte.to(DEVICE), yte.to(DEVICE)
    accs, params = [], None
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = build(image_size, patch_size)
        params = n_params(m)
        opt = optim.Adam(m.parameters(), lr=3e-3)
        N = Xtr.shape[0]
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(N, device=DEVICE)
            for i in range(0, N, bs_train):
                idx = perm[i:i + bs_train]
                opt.zero_grad()
                loss = F.nll_loss(F.log_softmax(m(Xtr[idx]), dim=1), ytr[idx])
                loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xte).argmax(1) == yte).float().mean().item()
        accs.append(acc * 100)
    return params, np.mean(accs), np.std(accs)


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    print(f"train {Xtr.shape[0]}  test {Xte.shape[0]}  (digits {CLASSES})\n")
    rows = []

    # baseline: full-res learned embedding
    p, mu, sd = run(Xtr, ytr, Xte, yte, image_size=28, patch_size=7)
    rows.append(("baseline 28x28 (learned 7x7 embed)", "784 px", 16, p, mu, sd, None))

    # tiny configs: (pool block size, resulting tiny size, tokens)
    for bs in (7, 4):
        tiny = 28 // bs
        for mode in ("mean", "median"):
            Xtr_p = pool(Xtr, bs, mode)
            Xte_p = pool(Xte, bs, mode)
            p, mu, sd = run(Xtr_p, ytr, Xte_p, yte, image_size=tiny, patch_size=1)
            rows.append((f"tiny {tiny}x{tiny} ({mode}-pool {bs}x{bs} patch)",
                         f"{tiny*tiny} px", tiny * tiny, p, mu, sd, mode))

    # report
    print(f"{'config':40s} {'pixels':>7s} {'tokens':>7s} {'params':>8s} {'accuracy':>16s}")
    print("-" * 84)
    for name, px, tok, p, mu, sd, _ in rows:
        print(f"{name:40s} {px:>7s} {tok:>7d} {p:>8,d}   {mu:>6.2f}% ± {sd:.2f}")


if __name__ == "__main__":
    main()
