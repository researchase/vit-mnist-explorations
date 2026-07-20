"""
Put the MLP back. Train a 1-block model WITH the feedforward MLP (attention THEN MLP, both with
residuals) on MNIST {0,1,2}, and trace what the MLP does to the CLS token — the stage we deleted
to get the clean "one axis per head" story.

Compares CLS separation (silhouette, all 4 dims) at each stage against the attention-only model,
and writes cls4d_mlp_data.js so cls_4d_mlp.html can show the journey with the MLP stage in 3D.
"""
import os, json
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from einops import repeat, rearrange
from sklearn.metrics import silhouette_score

import viz_journey as vj
from train_gpu import ViT                 # full ViT: Transformer = attention + FeedForward
from tiny_pool import load_tensors
from cls_4d import sample

DEVICE, HERE, CC = vj.DEVICE, vj.HERE, vj.CLASS_COLORS
CLASSES = vj.CLASSES


def build_mlp(mlp_dim=8):                 # 1 block, dim=4, 4 heads, patch=4, WITH MLP
    return ViT(image_size=28, patch_size=4, num_classes=3, channels=1,
               dim=4, depth=1, heads=4, dim_head=1, mlp_dim=mlp_dim).to(DEVICE)


def train(m, Xtr, ytr, Xte, yte, epochs=50):
    opt = optim.Adam(m.parameters(), lr=3e-3); N = Xtr.shape[0]
    for _ in range(epochs):
        m.train(); perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, 128):
            idx = perm[i:i+128]; opt.zero_grad()
            F.nll_loss(F.log_softmax(m(Xtr[idx]), 1), ytr[idx]).backward(); opt.step()
    m.eval()
    return (m(Xte).argmax(1) == yte).float().mean().item()


@torch.no_grad()
def trace_mlp(m, imgs):
    x = m.to_patch_embedding(imgs)
    b, n, _ = x.shape
    cls = repeat(m.cls_token, "() n d -> b n d", b=b)
    x = torch.cat([cls, x], 1); x = x + m.pos_embedding[:, :n+1]
    attn_pn, ff_pn = m.transformer.layers[0]
    at = attn_pn.fn
    normed = attn_pn.norm(x)
    q, k, v = at.to_qkv(normed).chunk(3, -1)
    qh, kh, vh = (rearrange(t, "b n (h d) -> b h n d", h=at.heads) for t in (q, k, v))
    A = (torch.einsum("bhid,bhjd->bhij", qh, kh) * at.scale).softmax(-1)
    out = torch.einsum("bhij,bhjd->bhid", A, vh)
    ocls = rearrange(out, "b h n d -> b n (h d)")[:, 0]     # [N,4]
    Wo, b0 = at.to_out[0].weight, at.to_out[0].bias
    contrib = [ocls[:, h:h+1] * Wo[:, h] for h in range(at.heads)]

    stages, names = [], []
    p = x[:, 0].clone(); stages.append(p.clone()); names.append("input · CLS+pos")
    p = p + b0;          stages.append(p.clone()); names.append("+ bias")
    for h in range(at.heads):
        p = p + contrib[h]; stages.append(p.clone()); names.append(f"+ head {h}")
    x = attn_pn(x) + x                                       # after attention (== p)
    x = ff_pn(x) + x
    stages.append(x[:, 0].clone()); names.append(">> after MLP <<")            # the new stage
    stages.append(m.mlp_head[0](x[:, 0])); names.append("final LayerNorm")
    return names, torch.stack(stages).cpu().numpy()


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    Xtr, ytr, Xte, yte = Xtr.to(DEVICE), ytr.to(DEVICE), Xte.to(DEVICE), yte.to(DEVICE)
    imgs, labels = sample(Xte, yte)

    print("== training 1-block, dim=4, 4 heads, patch=4, WITH MLP (mlp_dim=8) ==")
    torch.manual_seed(0); np.random.seed(0)
    m = build_mlp(8)
    acc = train(m, Xtr, ytr, Xte, yte, 50)
    nparam = sum(p.numel() for p in m.parameters())
    print(f"accuracy {acc*100:.2f}%   params {nparam}   (attention-only was 371 @ ~98.6%)\n")

    names, arr = trace_mlp(m, imgs)
    print(f"{'stage':22s} {'silhouette(4d)':>15s} {'Δ from prev':>12s}   cloud σ(size)")
    print("-"*64)
    prev = None
    sils = []
    for s, nm in enumerate(names):
        try: sv = silhouette_score(arr[s], labels)
        except Exception: sv = float("nan")
        sils.append(sv)
        d = "" if prev is None else f"{sv-prev:+.3f}"
        tot = float(np.trace(np.cov(arr[s].T)))
        print(f"{nm:22s} {sv:>15.3f} {d:>12s}   {tot:>7.2f}")
        prev = sv

    # what the MLP did: movement magnitude of CLS across the MLP step
    ia = names.index("+ head 3"); im = names.index(">> after MLP <<")
    mlp_move = np.linalg.norm(arr[im]-arr[ia], axis=1)
    print(f"\nMLP moved each CLS token by mean {mlp_move.mean():.3f} (max {mlp_move.max():.3f})")
    print(f"silhouette: after attention {sils[ia]:.3f}  ->  after MLP {sils[im]:.3f}  "
          f"({'MLP separates further' if sils[im]>sils[ia] else 'MLP does not add separation here'})")

    # export for the 3D viewer (attention arrows = to_out columns)
    Wo = m.transformer.layers[0][0].fn.to_out[0].weight.detach().cpu().numpy()
    data = dict(stages=names, coords=np.asarray(arr, float).round(4).tolist(),
                labels=[int(x) for x in labels],
                d3min=float(arr[..., 3].min()), d3max=float(arr[..., 3].max()),
                sil=[None if s != s else round(float(s), 3) for s in sils],
                headDirs=[Wo[:, h].tolist() for h in range(4)], classColors=CC)
    with open(os.path.join(HERE, "cls4d_mlp_data.js"), "w") as f:
        f.write("var CLS4D = "); json.dump(data, f); f.write(";\n")
    print("\nwrote cls4d_mlp_data.js")


if __name__ == "__main__":
    main()
