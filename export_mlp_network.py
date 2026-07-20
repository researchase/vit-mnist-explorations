"""
Export the WITH-MLP model (1 block, dim=4, 4 heads, patch=4, mlp_dim=8) to net_mlp_data.js,
same shape as net_data.js but with extra feedforward weights + activations, so the studio's
MLP toggle can run this model's live forward pass and render its extra sheets.
"""
import os, json
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets

import viz_journey as vj
from train_gpu import ViT
from tiny_pool import load_tensors, MEAN, STD

DEVICE, HERE, CLASSES = vj.DEVICE, vj.HERE, vj.CLASSES
L = lambda t: t.detach().cpu().numpy().astype(float).tolist()


@torch.no_grad()
def trace(m, img):
    x = m.to_patch_embedding(img)
    cls = m.cls_token
    emb = torch.cat([cls, x], 1)
    posed = emb + m.pos_embedding[:, :emb.shape[1]]
    attn_pn, ff_pn = m.transformer.layers[0]
    at = attn_pn.fn
    normed = attn_pn.norm(posed)
    q, k, v = at.to_qkv(normed).chunk(3, -1)
    qh, kh, vh = (rearrange(t, "b n (h d) -> b h n d", h=at.heads) for t in (q, k, v))
    A = (torch.einsum("bhid,bhjd->bhij", qh, kh) * at.scale).softmax(-1)
    out = torch.einsum("bhij,bhjd->bhid", A, vh)
    outcat = rearrange(out, "b h n d -> b n (h d)")
    delta = at.to_out(outcat)
    residA = posed + delta                              # after attention
    # MLP sublayer
    ff = ff_pn.fn
    mlpNorm = ff_pn.norm(residA)
    h1 = F.relu(ff.net[0](mlpNorm))                     # hidden after ReLU  [.,.,8]
    mlpDelta = ff.net[3](h1)
    residM = residA + mlpDelta                          # after MLP
    clsN = m.mlp_head[0](residM[:, 0])
    logits = m.mlp_head[1](clsN)
    return dict(embed=L(emb[0]), posed=L(posed[0]), normed=L(normed[0]),
                Q=L(q[0]), K=L(k[0]), V=L(v[0]), attn=L(A[0]),
                attnOut=L(outcat[0]), delta=L(delta[0]), resid=L(residA[0]),
                mlpNorm=L(mlpNorm[0]), mlpHidden=L(h1[0]), mlpDelta=L(mlpDelta[0]),
                residMlp=L(residM[0]), clsNorm=L(clsN[0]), logits=L(logits[0]), probs=L(logits.softmax(-1)[0]))


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    Xtr, ytr = Xtr.to(DEVICE), ytr.to(DEVICE)
    torch.manual_seed(0); np.random.seed(0)
    m = ViT(image_size=28, patch_size=4, num_classes=3, channels=1,
            dim=4, depth=1, heads=4, dim_head=1, mlp_dim=8).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=3e-3); N = Xtr.shape[0]
    for _ in range(50):
        m.train(); perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, 128):
            idx = perm[i:i+128]; opt.zero_grad()
            F.nll_loss(F.log_softmax(m(Xtr[idx]), 1), ytr[idx]).backward(); opt.step()
    m.eval()
    acc = (m(Xte.to(DEVICE)).argmax(1) == yte.to(DEVICE)).float().mean().item()
    nparam = sum(p.numel() for p in m.parameters())
    print(f"trained WITH MLP: {acc*100:.2f}%  params={nparam}")
    torch.save(m.state_dict(), os.path.join(HERE, "vit_455_mlp.pt"))

    attn_pn, ff_pn = m.transformer.layers[0]
    at, ff = attn_pn.fn, ff_pn.fn
    weights = dict(
        patchW=L(m.to_patch_embedding[1].weight), patchB=L(m.to_patch_embedding[1].bias),
        cls=L(m.cls_token[0, 0]), pos=L(m.pos_embedding[0]),
        lnW=L(attn_pn.norm.weight), lnB=L(attn_pn.norm.bias),
        qkvW=L(at.to_qkv.weight), outW=L(at.to_out[0].weight), outB=L(at.to_out[0].bias),
        mlpLnW=L(ff_pn.norm.weight), mlpLnB=L(ff_pn.norm.bias),
        mlpW1=L(ff.net[0].weight), mlpB1=L(ff.net[0].bias),
        mlpW2=L(ff.net[3].weight), mlpB2=L(ff.net[3].bias),
        headLnW=L(m.mlp_head[0].weight), headLnB=L(m.mlp_head[0].bias),
        clsW=L(m.mlp_head[1].weight), clsB=L(m.mlp_head[1].bias),
    )
    counts = {kk: int(np.array(v).size) for kk, v in weights.items()}
    print("param breakdown:", counts, "sum", sum(counts.values()))

    ds = datasets.MNIST("/mnt/nvme1tb/data/mnist", train=False, download=True)
    examples = []
    for c in CLASSES:
        i = int((ds.targets == c).nonzero()[0])
        px = ds.data[i]
        img = px.float().div(255).sub(MEAN).div(STD).view(1, 1, 28, 28).to(DEVICE)
        examples.append(dict(label=int(c), px=px.numpy().astype(int).reshape(-1).tolist(), act=trace(m, img)))

    data = dict(weights=weights, counts=counts, total=int(sum(counts.values())),
                acc=round(acc*100, 2), examples=examples, classColors=vj.CLASS_COLORS,
                cfg=dict(dim=4, heads=4, dimHead=1, tokens=50, patches=49, patchDim=16,
                         patch=4, grid=7, mlpDim=8))
    with open(os.path.join(HERE, "net_mlp_data.js"), "w") as f:
        f.write("var NET_MLP = "); json.dump(data, f); f.write(";\n")
    print("wrote net_mlp_data.js")


if __name__ == "__main__":
    main()
