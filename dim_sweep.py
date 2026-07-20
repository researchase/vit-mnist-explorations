"""
Sweep the embedding dim on the 1-block / 4-head attention-only model (full-res MNIST 0/1/2).
heads fixed at 4, dim_head = dim/4 (so attention inner = dim, standard ViT).
Question: is dim=64 needed, or does a smaller dim (far fewer params) match it?
"""
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

import viz_journey as vj
from train_gpu import ViT
from viz_journey_attn_only import AttnOnlyTransformer
from tiny_pool import load_tensors

DEVICE = vj.DEVICE


def build(dim, heads=4):
    dh = dim // heads
    m = ViT(image_size=28, patch_size=7, num_classes=3, channels=1,
            dim=dim, depth=1, heads=heads, dim_head=dh, mlp_dim=dim * 2).to(DEVICE)
    m.transformer = AttnOnlyTransformer(dim=dim, depth=1, heads=heads, dim_head=dh).to(DEVICE)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(Xtr, ytr, Xte, yte, dim, epochs=6, seeds=(0, 1, 2), bs=128):
    accs, params = [], None
    N = Xtr.shape[0]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = build(dim); params = n_params(m)
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
    Xtr, ytr, Xte, yte = Xtr.to(DEVICE), ytr.to(DEVICE), Xte.to(DEVICE), yte.to(DEVICE)
    print(f"train {Xtr.shape[0]} test {Xte.shape[0]}  (1 block, 4 heads, full-res)\n")
    print(f"{'dim':>5s} {'dim_head':>9s} {'params':>8s} {'accuracy':>16s}")
    print("-" * 42)
    for dim in (4, 8, 16, 32, 64, 128):
        p, mu, sd = run(Xtr, ytr, Xte, yte, dim)
        print(f"{dim:>5d} {dim//4:>9d} {p:>8,d}   {mu:>6.2f}% ± {sd:.2f}")


if __name__ == "__main__":
    main()
