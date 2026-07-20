"""
The CLS token is only 4-D in the 371-param model, so we can show ALL of it losslessly:
dims 0,1,2 -> x,y,z position; dim 3 -> marker diameter. No PCA, no information lost.
Journey uses the exact additive decomposition CLS_out = CLS_in + bias + Σ head_h, so the
point moves head-by-head, then a final LayerNorm frame (the classifier's actual input space).
"""
import os
import json
import numpy as np
import torch
from einops import repeat, rearrange
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

import viz_journey as vj
from min_model import build
from tiny_pool import load_tensors

DEVICE, HERE, CC = vj.DEVICE, vj.HERE, vj.CLASS_COLORS
CLASSES = vj.CLASSES


def sample(Xte, yte, per=150):
    xs, ys = [], []
    for c in CLASSES:
        idx = (yte == c).nonzero().flatten()[:per]
        xs.append(Xte[idx]); ys.append(np.full(len(idx), c))
    return torch.cat(xs).to(DEVICE), np.concatenate(ys)


@torch.no_grad()
def trace(model, imgs):
    x = model.to_patch_embedding(imgs)
    b, n, _ = x.shape
    cls = repeat(model.cls_token, "() n d -> b n d", b=b)
    x = torch.cat([cls, x], 1)
    x = x + model.pos_embedding[:, : n + 1]
    pn = model.transformer.layers[0]; at = pn.fn
    dh, heads = 1, at.heads
    normed = pn.norm(x)
    q, k, v = at.to_qkv(normed).chunk(3, -1)
    qh, kh, vh = (rearrange(t, "b n (h d) -> b h n d", h=heads) for t in (q, k, v))
    A = (torch.einsum("bhid,bhjd->bhij", qh, kh) * at.scale).softmax(-1)
    out = torch.einsum("bhij,bhjd->bhid", A, vh)
    outcat = rearrange(out, "b h n d -> b n (h d)")           # [N,17->50,4]
    Wo = at.to_out[0].weight; b0 = at.to_out[0].bias          # [4,4],[4]
    ocls = outcat[:, 0]                                        # [N,4]
    contrib = [ocls[:, h:h + 1] * Wo[:, h] for h in range(heads)]  # each [N,4]

    stages, names = [], []
    p = x[:, 0].clone(); stages.append(p.clone()); names.append("input · CLS + pos")
    p = p + b0;          stages.append(p.clone()); names.append("+ bias")
    for h in range(heads):
        p = p + contrib[h]; stages.append(p.clone()); names.append(f"+ head {h}")
    after = p.clone()                                          # == CLS after attention
    stages.append(model.mlp_head[0](after)); names.append("final LayerNorm (classifier input)")
    return names, torch.stack(stages).cpu().numpy()            # [S,N,4]


def build_html(names, arr, labels, out):
    S = arr.shape[0]
    lo = arr[..., :3].reshape(-1, 3).min(0); hi = arr[..., :3].reshape(-1, 3).max(0)
    pad = (hi - lo) * 0.08 + 1e-6
    d3 = arr[..., 3]; d3lo, d3hi = float(d3.min()), float(d3.max())   # dim-3 -> rainbow colour
    SYMB = ["circle", "diamond", "square"]

    frames, steps = [], []
    for s in range(S):
        data = []
        for ci, c in enumerate(CLASSES):
            m = labels == c
            data.append(go.Scatter3d(
                x=arr[s, m, 0], y=arr[s, m, 1], z=arr[s, m, 2],
                mode="markers", name=f"digit {c} ({SYMB[ci]})",
                marker=dict(size=5, symbol=SYMB[ci], opacity=0.85, line=dict(width=0),
                            color=arr[s, m, 3], colorscale="Jet", cmin=d3lo, cmax=d3hi,
                            showscale=(ci == 0),
                            colorbar=dict(title="dim 3", thickness=14, len=0.6)),
                hovertemplate=f"digit {c}<br>d0=%{{x:.2f}} d1=%{{y:.2f}} d2=%{{z:.2f}}<br>d3=%{{marker.color:.2f}}<extra></extra>"))
        frames.append(go.Frame(data=data, name=str(s)))
        try:
            sil4 = silhouette_score(arr[s], labels)
        except Exception:
            sil4 = float("nan")
        steps.append(dict(method="animate", label=names[s],
                          args=[[str(s)], dict(mode="immediate",
                                frame=dict(duration=0, redraw=True), transition=dict(duration=0))]))
        frames[-1].layout = go.Layout(
            title=f"CLS in raw 4-D — {names[s]}  ·  silhouette(all 4 dims)={sil4:.3f}")

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=f"CLS in raw 4-D — {names[0]}",
        scene=dict(xaxis=dict(title="dim 0", range=[lo[0]-pad[0], hi[0]+pad[0]]),
                   yaxis=dict(title="dim 1", range=[lo[1]-pad[1], hi[1]+pad[1]]),
                   zaxis=dict(title="dim 2", range=[lo[2]-pad[2], hi[2]+pad[2]])),
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="stage: "))],
        updatemenus=[dict(type="buttons", y=1.05, x=0,
                     buttons=[dict(label="▶ play", method="animate",
                              args=[None, dict(frame=dict(duration=650, redraw=True),
                                    fromcurrent=True, transition=dict(duration=250))])])],
        annotations=[dict(showarrow=False, x=0, y=0, xref="paper", yref="paper",
                     text="colour = dim 3 (blue→red)  ·  shape = class  ·  all four dims shown, no projection",
                     font=dict(color="#888"))],
        margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(out, include_plotlyjs=True, auto_open=False)
    print("wrote", out)


def write_data(names, arr, labels, sils, head_dirs):
    data = dict(stages=names, coords=np.asarray(arr, float).round(4).tolist(),
                labels=[int(x) for x in labels],
                d3min=float(arr[..., 3].min()), d3max=float(arr[..., 3].max()),
                sil=[None if s != s else round(float(s), 3) for s in sils],
                headDirs=np.asarray(head_dirs, float).round(4).tolist(),
                classColors=CC)
    with open(os.path.join(HERE, "cls4d_data.js"), "w") as f:
        f.write("var CLS4D = "); json.dump(data, f); f.write(";\n")
    print("wrote cls4d_data.js")


def main():
    model = build(28, 4, 4, 4)
    model.load_state_dict(torch.load(os.path.join(HERE, "vit_371.pt"), map_location=DEVICE))
    model.eval()
    (_, _), (Xte, yte) = load_tensors()
    imgs, labels = sample(Xte, yte)
    names, arr = trace(model, imgs)
    print("stages:", names)
    sils = []
    for s, nm in enumerate(names):
        try:
            sv = silhouette_score(arr[s], labels)
        except Exception:
            sv = float("nan")
        sils.append(sv)
        print(f"  {nm:38s} silhouette(4d)={sv:.3f}")
    build_html(names, arr, labels, os.path.join(HERE, "cls_4d_journey.html"))
    W = model.transformer.layers[0].fn.to_out[0].weight.detach().cpu().numpy()  # [4,4]
    head_dirs = [W[:, h].tolist() for h in range(W.shape[1])]                    # column h = head h axis
    write_data(names, arr, labels, sils, head_dirs)


if __name__ == "__main__":
    main()
