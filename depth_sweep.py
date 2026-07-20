"""
Is "block 1 with 4 heads" enough for MNIST {0,1,2}?  Tests two ways:
  (A) Ablate the trained 4-block attention-only model: disable blocks 2..k -> identity.
  (B) Fair retrain: train fresh attention-only models at depth = 1,2,3,4 (multi-seed).
  (C) Heads sweep: depth=1 with heads = 1,2,4.
"""
import os
import numpy as np
import torch

import viz_journey as vj
from train_gpu import ViT
from viz_journey_attn_only import AttnOnlyTransformer
from viz_heads import replay

DEVICE = vj.DEVICE
HERE = vj.HERE


def build(depth, heads=4):
    m = ViT(image_size=28, patch_size=7, num_classes=3, channels=1,
            dim=64, depth=1, heads=heads, dim_head=16, mlp_dim=128).to(DEVICE)
    m.transformer = AttnOnlyTransformer(dim=64, depth=depth, heads=heads, dim_head=16).to(DEVICE)
    return m


@torch.no_grad()
def eval_first_k_blocks(model, loader, k):
    """Evaluate using only the first k attention blocks (rest = identity)."""
    model.eval()
    correct = total = 0
    for data, target in loader:
        off = frozenset()  # we implement 'identity' by truncating the block loop below
        data = data.to(DEVICE)
        # manual forward through first k blocks
        from einops import repeat
        x = model.to_patch_embedding(data)
        b, n, _ = x.shape
        cls = repeat(model.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls, x), dim=1) + model.pos_embedding[:, : n + 1]
        for bi, pn in enumerate(model.transformer.layers):
            if bi >= k:
                break
            x = pn(x) + x
        logits = model.mlp_head(x[:, 0])
        correct += (logits.argmax(1) == target.to(DEVICE)).sum().item()
        total += target.numel()
    return correct / total


def main():
    train_loader, test_loader = vj.make_loaders()

    # ---- (A) ablate the existing trained 4-block model ----
    print("== (A) trained 4-block model, using only first k blocks (rest identity) ==")
    m4 = build(4)
    m4.load_state_dict(torch.load(os.path.join(HERE, "vit_mnist_3class_attn_only.pt"),
                                  map_location=DEVICE))
    for k in range(1, 5):
        print(f"   first {k} block(s): acc {eval_first_k_blocks(m4, test_loader, k)*100:.2f}%")

    # ---- (B) fair retrain at each depth, 3 seeds ----
    print("\n== (B) train fresh attention-only models at each depth (3 seeds) ==")
    seeds = [0, 1, 2]
    for depth in [1, 2, 3, 4]:
        accs = []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            m = build(depth)
            vj.train_model(m, train_loader, test_loader, epochs=4)  # prints per-epoch
            accs.append(vj.evaluate(m, test_loader))
        accs = np.array(accs) * 100
        print(f"   depth={depth}: acc {accs.mean():.2f}% ± {accs.std():.2f}  (seeds: "
              f"{', '.join(f'{a:.2f}' for a in accs)})\n")

    # ---- (C) heads sweep at depth=1 ----
    print("== (C) depth=1, heads sweep (3 seeds) ==")
    for heads in [1, 2, 4]:
        accs = []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            m = build(1, heads=heads)
            vj.train_model(m, train_loader, test_loader, epochs=4)
            accs.append(vj.evaluate(m, test_loader))
        accs = np.array(accs) * 100
        print(f"   1 block, {heads} head(s): acc {accs.mean():.2f}% ± {accs.std():.2f}")


if __name__ == "__main__":
    main()
