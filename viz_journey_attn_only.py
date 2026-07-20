"""
Attention-ONLY ViT vs full ViT on MNIST {0,1,2}.

Controlled A/B test of the "attention does the separating, MLP refines" claim:
  - FULL model:      each block = attention(+residual) THEN feedforward MLP(+residual)
  - ATTN-ONLY model: each block = attention(+residual) only  (feedforward removed)

Both share the same patch embedding, CLS token, positional embedding, and the final
single-Linear classification head (kept -- it's the classifier, not a channel-mixing MLP).
Same data, same sampled images, same seed -> a fair head-to-head.

Reuses ViT / PreNorm / Attention from train_gpu.py and the helpers in viz_journey.py.
"""
import os
import numpy as np
import torch
from torch import nn
from einops import repeat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_gpu import ViT, PreNorm, Attention
import viz_journey as vj  # make_loaders, sample_images, train_model, evaluate, project_all, ...

HERE = vj.HERE
DEVICE = vj.DEVICE
CLASSES = vj.CLASSES

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------------------------------------------
# Attention-only transformer: drop the FeedForward sub-layer entirely.
# --------------------------------------------------------------------------------------
class AttnOnlyTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x
        return x


def make_full_model():
    return ViT(image_size=28, patch_size=7, num_classes=len(CLASSES), channels=1,
               dim=64, depth=4, heads=4, dim_head=16, mlp_dim=128).to(DEVICE)


def make_attn_only_model():
    m = make_full_model()
    # swap the full transformer stack for the attention-only one
    m.transformer = AttnOnlyTransformer(dim=64, depth=4, heads=4, dim_head=16).to(DEVICE)
    return m


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Journeys (CLS token after each residual add). Full model has 2 adds/block; attn-only 1.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def trace_full(model, imgs):
    return vj.trace_cls(model, imgs)  # 1 + 2*depth stages


@torch.no_grad()
def trace_attn_only(model, imgs):
    model.eval()
    imgs = imgs.to(DEVICE)
    x = model.to_patch_embedding(imgs)
    b, n, _ = x.shape
    cls = repeat(model.cls_token, "() n d -> b n d", b=b)
    x = torch.cat((cls, x), dim=1)
    x = x + model.pos_embedding[:, : (n + 1)]

    stages, names = [x[:, 0].clone()], ["S0: input\n(patch+CLS+pos)"]
    for i, attn in enumerate(model.transformer.layers, start=1):
        x = attn(x) + x
        stages.append(x[:, 0].clone())
        names.append(f"block {i}\nafter attention")
    arr = torch.stack(stages, 0).cpu().numpy()  # [1+depth, N, dim]
    return names, arr


def per_block_silhouette_full(arr, labels):
    """End-of-block values for the full model: S0 + each block's 'after MLP' stage."""
    idx = [0] + [2 * b for b in range(1, arr.shape[0] // 2 + 1)]  # 0,2,4,6,8
    sil = vj.stage_silhouettes(arr, labels)
    return [sil[i] for i in idx]


def per_block_silhouette_attn(arr, labels):
    return vj.stage_silhouettes(arr, labels)  # already one value per block + S0


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = vj.make_loaders()
    imgs, labels = vj.sample_images(test_loader, per_class=200)
    print(f"Sample: {len(labels)} images {CLASSES}\n")

    # ---- Full model (attention + MLP) ----
    print("== Training FULL model (attention + MLP) ==")
    full = make_full_model()
    vj.train_model(full, train_loader, test_loader, epochs=5)
    full_acc = vj.evaluate(full, test_loader)

    # ---- Attention-only model ----
    print("\n== Training ATTENTION-ONLY model (no MLP) ==")
    attn = make_attn_only_model()
    vj.train_model(attn, train_loader, test_loader, epochs=5)
    attn_acc = vj.evaluate(attn, test_loader)
    torch.save(attn.state_dict(), os.path.join(HERE, "vit_mnist_3class_attn_only.pt"))

    # ---- Journeys ----
    full_names, full_arr = trace_full(full, imgs)
    attn_names, attn_arr = trace_attn_only(attn, imgs)

    full_proj, _ = vj.project_all(full_arr)
    attn_proj, _ = vj.project_all(attn_arr)
    full_sil_all = vj.stage_silhouettes(full_arr, labels)
    attn_sil_all = vj.stage_silhouettes(attn_arr, labels)

    # attention-only visual artifacts
    vj.build_html(attn_proj, labels, attn_names, attn_sil_all,
                  os.path.join(HERE, "attn_only_journey.html"),
                  "ATTENTION-ONLY ViT (no MLP) — CLS journey")
    vj.build_grid_png(attn_proj, labels, attn_names, attn_sil_all,
                      os.path.join(HERE, "attn_only_journey_grid.png"),
                      "ATTENTION-ONLY ViT (no MLP)")

    # per-block comparison line chart
    full_block = per_block_silhouette_full(full_arr, labels)
    attn_block = per_block_silhouette_attn(attn_arr, labels)
    xs = list(range(len(full_block)))  # 0..4  (S0, blk1..4)
    plt.figure(figsize=(7, 5))
    plt.plot(xs, full_block, "o-", label=f"full (attn+MLP)  acc={full_acc*100:.2f}%")
    plt.plot(xs, attn_block, "s--", label=f"attention-only     acc={attn_acc*100:.2f}%")
    plt.xticks(xs, ["input"] + [f"blk{b}" for b in range(1, len(xs))])
    plt.xlabel("depth (end of block)")
    plt.ylabel("silhouette score of 3 classes")
    plt.title("Class separation by depth: full vs attention-only")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    cmp_path = os.path.join(HERE, "compare_full_vs_attn_only.png")
    plt.savefig(cmp_path, dpi=120)
    plt.close()

    # ---- Report ----
    print("\n" + "=" * 60)
    print("RESULT: full (attention+MLP)  vs  attention-only (no MLP)")
    print("=" * 60)
    print(f"  params:            {n_params(full):>8,d}   {n_params(attn):>8,d}   "
          f"(MLP removed: {n_params(full)-n_params(attn):,d})")
    print(f"  test accuracy:     {full_acc*100:>7.2f}%   {attn_acc*100:>7.2f}%")
    print(f"  final silhouette:  {full_sil_all[-1]:>8.3f}   {attn_sil_all[-1]:>8.3f}")
    print("\n  per-block silhouette:")
    print(f"    {'stage':10s} {'full':>8s} {'attn-only':>10s}")
    for b in range(len(full_block)):
        nm = "input" if b == 0 else f"blk{b}"
        print(f"    {nm:10s} {full_block[b]:>8.3f} {attn_block[b]:>10.3f}")
    print(f"\n  wrote {cmp_path}")
    print("  wrote attn_only_journey.html / attn_only_journey_grid.png")


if __name__ == "__main__":
    main()
