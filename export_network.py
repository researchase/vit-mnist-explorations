"""
Train the 371-parameter model (full-res, dim=4, 4 heads, 1 block, no MLP) and export every weight
plus per-example activations to `net_data.js` for the 3D bbycroft-style visualization.
"""
import os, json
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets

import viz_journey as vj
from min_model import build
from tiny_pool import load_tensors, MEAN, STD

DEVICE = vj.DEVICE
HERE = vj.HERE
CLASSES = vj.CLASSES


def L(t):
    return t.detach().cpu().numpy().astype(float).tolist()


@torch.no_grad()
def trace(m, img):                       # img: [1,1,28,28]
    x = m.to_patch_embedding(img)        # [1,16,4]
    cls = m.cls_token                    # [1,1,4]
    emb = torch.cat([cls, x], 1)         # [1,T,4]
    posed = emb + m.pos_embedding[:, :emb.shape[1]]
    pn = m.transformer.layers[0]
    normed = pn.norm(posed)
    at = pn.fn
    q, k, v = at.to_qkv(normed).chunk(3, -1)
    qh, kh, vh = (rearrange(t, "b n (h d) -> b h n d", h=at.heads) for t in (q, k, v))
    dots = torch.einsum("b h i d, b h j d -> b h i j", qh, kh) * at.scale
    A = dots.softmax(-1)                 # [1,4,17,17]
    out = torch.einsum("b h i j, b h j d -> b h i d", A, vh)
    outcat = rearrange(out, "b h n d -> b n (h d)")
    delta = at.to_out(outcat)
    resid = posed + delta
    clsN = m.mlp_head[0](resid[:, 0])
    logits = m.mlp_head[1](clsN)
    probs = logits.softmax(-1)
    return dict(embed=L(emb[0]), posed=L(posed[0]), normed=L(normed[0]),
                Q=L(q[0]), K=L(k[0]), V=L(v[0]), attn=L(A[0]),
                attnOut=L(outcat[0]), delta=L(delta[0]), resid=L(resid[0]),
                clsNorm=L(clsN[0]), logits=L(logits[0]), probs=L(probs[0]))


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    torch.manual_seed(0); np.random.seed(0)
    m = build(28, 4, 4, 4)               # patch_size=4 -> 7x7 grid, 49 patches + CLS = 50 tokens
    opt = optim.Adam(m.parameters(), lr=3e-3)
    N = Xtr.shape[0]
    for ep in range(50):
        m.train(); perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            F.nll_loss(F.log_softmax(m(Xtr[idx]), 1), ytr[idx]).backward(); opt.step()
    m.eval()
    acc = (m(Xte.to(DEVICE)).argmax(1) == yte.to(DEVICE)).float().mean().item()
    nparam = sum(p.numel() for p in m.parameters())
    print(f"trained: {acc*100:.2f}%  params={nparam}")
    torch.save(m.state_dict(), os.path.join(HERE, "vit_371.pt"))

    at = m.transformer.layers[0].fn
    weights = dict(
        patchW=L(m.to_patch_embedding[1].weight), patchB=L(m.to_patch_embedding[1].bias),
        cls=L(m.cls_token[0, 0]), pos=L(m.pos_embedding[0]),
        lnW=L(m.transformer.layers[0].norm.weight), lnB=L(m.transformer.layers[0].norm.bias),
        qkvW=L(at.to_qkv.weight), outW=L(at.to_out[0].weight), outB=L(at.to_out[0].bias),
        headLnW=L(m.mlp_head[0].weight), headLnB=L(m.mlp_head[0].bias),
        clsW=L(m.mlp_head[1].weight), clsB=L(m.mlp_head[1].bias),
    )
    counts = {k: int(np.array(v).size) for k, v in weights.items()}
    print("param breakdown:", counts, "sum", sum(counts.values()))

    # example activations (one 0, one 1, one 2)
    ds = datasets.MNIST("/mnt/nvme1tb/data/mnist", train=False, download=True)
    examples = []
    for c in CLASSES:
        i = int((ds.targets == c).nonzero()[0])
        px = ds.data[i]
        img = px.float().div(255).sub(MEAN).div(STD).view(1, 1, 28, 28).to(DEVICE)
        act = trace(m, img)
        examples.append(dict(label=int(c), px=px.numpy().astype(int).reshape(-1).tolist(), act=act))

    data = dict(weights=weights, counts=counts, total=int(sum(counts.values())),
                acc=round(acc * 100, 2), examples=examples, classColors=vj.CLASS_COLORS,
                cfg=dict(dim=4, heads=4, dimHead=1, tokens=50, patches=49, patchDim=16,
                         patch=4, grid=7))
    with open(os.path.join(HERE, "net_data.js"), "w") as f:
        f.write("var NET = "); json.dump(data, f); f.write(";\n")
    print("wrote net_data.js", f"{os.path.getsize(os.path.join(HERE,'net_data.js'))/1024:.0f} KB")


if __name__ == "__main__":
    main()
