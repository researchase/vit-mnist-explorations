"""
Head-by-head dissection of the ATTENTION-ONLY ViT on MNIST {0,1,2}.

Because the block has no MLP, the residual update to every token is exactly:
    x_out = x_in + to_out( concat_h [ attention_h(x_in) ] )
and to_out is a single Linear, so for the CLS token this is a clean linear sum:
    CLS_out = CLS_in + bias + Σ_h contribution_h
where contribution_h = (head h's CLS output, 16-dim) @ (its slice of to_out, 16->64).

This lets us:
  1. Draw, per block, the chain of 4 per-head displacement arrows moving the CLS token
     (fig: heads_cls_movement.png) -- the direct "how each head moves the CLS token".
  2. Show WHERE each head's CLS query looks (attention over the 16 patches)
     (fig: heads_attention_maps.png).
  3. Rank heads by (a) how far they move the CLS token and (b) knockout impact on accuracy
     (fig: heads_importance.png + printed table).

Loads the trained weights saved by viz_journey_attn_only.py (no retraining).
"""
import os
import numpy as np
import torch
from einops import rearrange, repeat
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_journey as vj
from viz_journey_attn_only import make_attn_only_model

HERE = vj.HERE
DEVICE = vj.DEVICE
CLASSES = vj.CLASSES
CLASS_COLORS = vj.CLASS_COLORS
HEAD_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]  # 4 heads

torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------------------
# Manual forward that (a) advances the residual stream and (b) captures, per block:
#   cls_in, cls_out, per-head contribution vectors, per-head CLS->patch attention, bias.
# Optional knockout: a set of (block, head) whose output is zeroed everywhere.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def replay(model, imgs, knockout=frozenset()):
    model.eval()
    imgs = imgs.to(DEVICE)
    x = model.to_patch_embedding(imgs)          # [N, 16, dim]
    b, n, _ = x.shape
    cls = repeat(model.cls_token, "() n d -> b n d", b=b)
    x = torch.cat((cls, x), dim=1)              # [N, 17, dim]
    x = x + model.pos_embedding[:, : (n + 1)]

    heads = model.transformer.layers[0].fn.heads
    inner = model.transformer.layers[0].fn.to_qkv.out_features // 3
    dh = inner // heads

    cap = {k: [] for k in ("cls_in", "cls_out", "contrib", "attn_cls", "bias")}
    for bi, prenorm in enumerate(model.transformer.layers):
        attn_m = prenorm.fn
        normed = prenorm.norm(x)
        qkv = attn_m.to_qkv(normed).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), qkv)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * attn_m.scale
        attn = dots.softmax(dim=-1)             # [N, heads, 17, 17]
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)  # [N, heads, 17, dh]

        for (kb, kh) in knockout:               # ablate heads
            if kb == bi:
                out[:, kh] = 0

        out_cat = rearrange(out, "b h n d -> b n (h d)")     # [N, 17, inner]
        W = attn_m.to_out[0].weight             # [dim, inner]
        b0 = attn_m.to_out[0].bias              # [dim]
        delta = out_cat @ W.t() + b0            # [N, 17, dim]  (dropout=identity in eval)

        # per-head contribution to the CLS (i=0) update
        o_cls = out[:, :, 0, :]                 # [N, heads, dh]
        contrib = torch.stack(
            [o_cls[:, h] @ W[:, h * dh:(h + 1) * dh].t() for h in range(heads)], dim=1
        )                                       # [N, heads, dim]

        cap["cls_in"].append(x[:, 0].cpu().numpy())
        cap["contrib"].append(contrib.cpu().numpy())
        cap["attn_cls"].append(attn[:, :, 0, 1:].cpu().numpy())  # CLS->16 patches
        cap["bias"].append(b0.cpu().numpy())
        x = x + delta
        cap["cls_out"].append(x[:, 0].cpu().numpy())

    logits = model.mlp_head(x[:, 0])
    for kk in ("cls_in", "cls_out", "contrib", "attn_cls"):
        cap[kk] = np.stack(cap[kk], axis=1)     # [N, depth, ...]
    cap["bias"] = np.stack(cap["bias"], axis=0)  # [depth, dim]
    return logits.cpu(), cap


# sanity: Σ_h contrib + bias == cls_out - cls_in
def _check(cap):
    recon = cap["contrib"].sum(axis=2) + cap["bias"][None]  # [N, depth, dim]
    true = cap["cls_out"] - cap["cls_in"]
    err = np.abs(recon - true).max()
    print(f"  decomposition max error: {err:.2e}  (should be ~0)")


# --------------------------------------------------------------------------------------
# Fig 1: per-block chain of per-head displacement arrows moving the CLS token.
# --------------------------------------------------------------------------------------
def fig_movement(cap, labels, out_path):
    depth = cap["cls_in"].shape[1]
    heads = cap["contrib"].shape[2]
    pca = PCA(n_components=3, random_state=0).fit(
        np.concatenate([cap["cls_in"].reshape(-1, cap["cls_in"].shape[-1]),
                        cap["cls_out"].reshape(-1, cap["cls_in"].shape[-1])], axis=0)
    )

    def proj(v):
        return pca.transform(v.reshape(-1, v.shape[-1])).reshape(*v.shape[:-1], 3)

    fig = plt.figure(figsize=(14, 12))
    for bi in range(depth):
        ax = fig.add_subplot(2, 2, bi + 1, projection="3d")
        cin = proj(cap["cls_in"][:, bi])        # [N,3]
        for ci, c in enumerate(CLASSES):
            m = labels == c
            ax.scatter(cin[m, 0], cin[m, 1], cin[m, 2], s=5,
                       c=CLASS_COLORS[ci], alpha=0.20)
            # class centroid chain: in -> +bias -> +head0 -> ... -> out
            p = cap["cls_in"][m, bi].mean(0)
            pts = [p.copy()]
            p = p + cap["bias"][bi]
            pts.append(p.copy())               # after bias
            for h in range(heads):
                p = p + cap["contrib"][m, bi, h].mean(0)
                pts.append(p.copy())
            pts = pca.transform(np.stack(pts))  # [heads+2, 3]
            # bias segment (gray), then one colored segment per head
            ax.plot(pts[0:2, 0], pts[0:2, 1], pts[0:2, 2], color="0.6", lw=1.0, ls=":")
            for h in range(heads):
                seg = pts[h + 1:h + 3]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        color=HEAD_COLORS[h], lw=2.5,
                        label=(f"head {h}" if (bi == 0 and ci == 0) else None))
            ax.scatter(*pts[-1], c=CLASS_COLORS[ci], s=60, edgecolors="k", marker="*")
        ax.set_title(f"block {bi + 1}: per-head CLS displacement", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    handles, lbls = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=heads, fontsize=10)
    fig.suptitle("How each head moves the CLS token (arrows = class-mean displacement per head)\n"
                 "dotted gray = bias, ★ = block output centroid", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Fig 2: where each head's CLS query looks (attention over 16 patches), block 1.
# --------------------------------------------------------------------------------------
def fig_attention_maps(cap, imgs, labels, out_path, block=0, per_class=2):
    heads = cap["attn_cls"].shape[2]
    idxs = []
    for c in CLASSES:
        idxs += list(np.where(labels == c)[0][:per_class])
    imgs_np = imgs.cpu().numpy()

    rows = len(idxs)
    cols = 1 + heads
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    for r, i in enumerate(idxs):
        img = imgs_np[i, 0] * 0.3081 + 0.1307   # un-normalize for display
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_ylabel(f"digit {labels[i]}", fontsize=9)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("input", fontsize=9)
        for h in range(heads):
            amap = cap["attn_cls"][i, block, h].reshape(4, 4)
            amap = np.kron(amap, np.ones((7, 7)))  # 4x4 -> 28x28 (patch upsample)
            ax = axes[r, h + 1]
            ax.imshow(img, cmap="gray")
            ax.imshow(amap, cmap="jet", alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"head {h}", fontsize=9)
    fig.suptitle(f"Where each head's CLS query attends (block {block + 1})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Fig 3 + table: per-head displacement magnitude & knockout impact on accuracy.
# --------------------------------------------------------------------------------------
def head_importance(model, cap, labels, test_loader, out_path):
    depth = cap["contrib"].shape[1]
    heads = cap["contrib"].shape[2]

    # mean displacement magnitude each head applies to CLS
    mag = np.linalg.norm(cap["contrib"], axis=-1).mean(axis=0)  # [depth, heads]

    # baseline accuracy
    def acc_with(knock):
        correct = total = 0
        for data, target in test_loader:
            logits, _ = replay(model, data, knockout=knock)
            correct += (logits.argmax(1) == target).sum().item()
            total += target.numel()
        return correct / total

    base_acc = acc_with(frozenset())
    drop = np.zeros((depth, heads))
    for bi in range(depth):
        for h in range(heads):
            drop[bi, h] = base_acc - acc_with(frozenset({(bi, h)}))

    # figure: two heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title, fmt in (
        (axes[0], mag, "mean CLS displacement magnitude", "%.2f"),
        (axes[1], drop * 100, "accuracy drop when head removed (pp)", "%.2f"),
    ):
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_xticks(range(heads)); ax.set_xticklabels([f"h{h}" for h in range(heads)])
        ax.set_yticks(range(depth)); ax.set_yticklabels([f"block {b+1}" for b in range(depth)])
        ax.set_title(title, fontsize=11)
        for bi in range(depth):
            for h in range(heads):
                ax.text(h, bi, fmt % data[bi, h], ha="center", va="center",
                        color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Per-head importance (baseline accuracy {base_acc*100:.2f}%)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")

    # printed table
    print(f"\n  baseline accuracy: {base_acc*100:.2f}%")
    print(f"  {'block':7s} {'head':5s} {'displacement':>13s} {'acc_drop(pp)':>13s}")
    order = sorted(((bi, h) for bi in range(depth) for h in range(heads)),
                   key=lambda t: drop[t], reverse=True)
    for bi, h in order:
        print(f"  block {bi+1:<2d}   h{h:<3d} {mag[bi,h]:>13.3f} {drop[bi,h]*100:>13.2f}")


def main():
    print(f"Device: {DEVICE}")
    weights = os.path.join(HERE, "vit_mnist_3class_attn_only.pt")
    model = make_attn_only_model()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    print(f"Loaded {weights}")

    _, test_loader = vj.make_loaders()
    imgs, labels = vj.sample_images(test_loader, per_class=200)

    logits, cap = replay(model, imgs)
    _check(cap)
    print(f"  heads/block: {cap['contrib'].shape[2]}   blocks: {cap['contrib'].shape[1]}")

    fig_movement(cap, labels, os.path.join(HERE, "heads_cls_movement.png"))
    fig_attention_maps(cap, imgs, labels, os.path.join(HERE, "heads_attention_maps.png"), block=0)
    head_importance(model, cap, labels, test_loader,
                    os.path.join(HERE, "heads_importance.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
