"""
Export the trained attention-only ViT (weights + PCA basis + example images) to
`he_data.js`, so the whole forward pass can be re-run live in the browser and heads/blocks
can be toggled interactively. Run after viz_journey_attn_only.py has produced the weights.
"""
import os
import json
import numpy as np
import torch

import viz_journey as vj
from viz_journey_attn_only import make_attn_only_model
from viz_heads import replay
from sklearn.decomposition import PCA

HERE = vj.HERE
DEVICE = vj.DEVICE
CLASSES = vj.CLASSES


def as_list(t):
    return t.detach().cpu().numpy().astype(np.float64).tolist()


def main():
    model = make_attn_only_model()
    model.load_state_dict(torch.load(os.path.join(HERE, "vit_mnist_3class_attn_only.pt"),
                                     map_location=DEVICE))
    model.eval()

    # ---- weights ----
    lin = model.to_patch_embedding[1]           # Linear(49->64)
    blocks = []
    for pn in model.transformer.layers:         # PreNorm(Attention)
        attn = pn.fn
        blocks.append(dict(
            lnW=as_list(pn.norm.weight), lnB=as_list(pn.norm.bias), eps=pn.norm.eps,
            qkvW=as_list(attn.to_qkv.weight),          # [192,64], no bias
            outW=as_list(attn.to_out[0].weight),       # [64,64]
            outB=as_list(attn.to_out[0].bias),         # [64]
            scale=float(attn.scale),
        ))
    head = dict(
        lnW=as_list(model.mlp_head[0].weight), lnB=as_list(model.mlp_head[0].bias),
        eps=model.mlp_head[0].eps,
        linW=as_list(model.mlp_head[1].weight), linB=as_list(model.mlp_head[1].bias),
    )
    cfg = dict(dim=64, heads=model.transformer.layers[0].fn.heads, dimHead=16,
               depth=len(blocks), patch=7, grid=4)

    # ---- PCA basis fit on the CLS trajectory (cls_in & cls_out across blocks) ----
    _, test_loader = vj.make_loaders()
    imgs, labels = vj.sample_images(test_loader, per_class=200)
    _, cap = replay(model, imgs)
    dim = cap["cls_in"].shape[-1]
    fitset = np.concatenate([cap["cls_in"].reshape(-1, dim),
                             cap["cls_out"].reshape(-1, dim)], axis=0)
    pca = PCA(n_components=3, random_state=0).fit(fitset)

    def proj(v):
        return pca.transform(v.reshape(-1, dim)).reshape(*v.shape[:-1], 3)

    finals = cap["cls_out"][:, -1]              # [N, dim]
    centroids = []
    for c in CLASSES:
        p = proj(finals[labels == c].mean(0, keepdims=True))[0]
        centroids.append(dict(label=int(c), p=[float(z) for z in p]))

    # ---- example images (raw uint8 for display; normalized in JS for the forward) ----
    test_raw = torch.utils.data.DataLoader  # noqa (silence lint)
    from torchvision import datasets
    ds = datasets.MNIST(vj.DATA_PATH if hasattr(vj, "DATA_PATH") else "/mnt/nvme1tb/data/mnist",
                        train=False, download=True)
    examples = []
    per_class = 4
    counts = {c: 0 for c in CLASSES}
    for i in range(len(ds)):
        lab = int(ds.targets[i])
        if lab in counts and counts[lab] < per_class:
            px = ds.data[i].numpy().astype(np.uint8).reshape(-1).tolist()
            examples.append(dict(label=lab, px=px))
            counts[lab] += 1
        if all(counts[c] >= per_class for c in CLASSES):
            break
    # order examples 0,0,0,0,1,1,1,1,2,2,2,2
    examples.sort(key=lambda e: e["label"])

    data = dict(cfg=cfg,
                patchW=as_list(lin.weight), patchB=as_list(lin.bias),
                cls=as_list(model.cls_token.reshape(-1)),
                pos=as_list(model.pos_embedding.reshape(model.pos_embedding.shape[1], -1)),
                blocks=blocks, head=head,
                pca=dict(mean=[float(z) for z in pca.mean_],
                         comp=[[float(z) for z in row] for row in pca.components_]),
                centroids=centroids, examples=examples,
                classColors=vj.CLASS_COLORS)

    out = os.path.join(HERE, "he_data.js")
    with open(out, "w") as f:
        f.write("var DATA = ")
        json.dump(data, f)
        f.write(";\n")
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")

    # also emit PyTorch predictions for the examples so we can verify the JS forward
    ex_imgs = []
    for e in examples:
        arr = (np.array(e["px"], dtype=np.float32).reshape(1, 28, 28) / 255.0 - 0.1307) / 0.3081
        ex_imgs.append(arr)
    ex_t = torch.tensor(np.stack(ex_imgs)).to(DEVICE)
    with torch.no_grad():
        pred = model(ex_t).argmax(1).cpu().numpy().tolist()
    ver = [{"label": e["label"], "pred": p} for e, p in zip(examples, pred)]
    with open(os.path.join(HERE, "he_expected.json"), "w") as f:
        json.dump(ver, f)
    print("expected preds:", ver)


if __name__ == "__main__":
    main()
