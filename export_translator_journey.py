"""Export data for translator_journey.html — 3D/4D visualization of the
dim=8/4-head english->kannada translator.

Spaces exported:
1. Encoder token journey: 8-D token states PCA'd to 4 (x,y,z + color=PC4),
   at three stages: patch embed / +position / after attention.
2. Per-head views: each head is exactly 2-D (dh=2), so q/k/v per head are
   plotted directly — no collapsing.
3. Decoder word space: the 64 output-word directions (rows of the head
   matrix) PCA'd to 4, plus the greedy generation trajectory of the decoder
   hidden state through that space, one per input class. Also the input-side
   vocabulary embedding cloud (65 words incl. BOS).
"""
import json
import numpy as np
import torch

from tiny_pool import load_tensors
import viz_journey as vj
from translator import Translator, BOS, SEQ, K, P

DEVICE = vj.DEVICE
DIM, HEADS = 8, 4
CLS_NAMES = ["0", "1", "2"]


def pca4(X):
    """X: N,D -> N,4 projection, plus variance explained."""
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2 / (S ** 2).sum())[:4].tolist()
    return Xc @ Vh[:4].T, Vh[:4], mu, var


def project(X, Vh, mu):
    return (X - mu) @ Vh.T


def rl(t):  # round list for compact json
    return [[round(v, 4) for v in row] for row in t.tolist()]


def main():
    m = Translator(DIM, HEADS).to(DEVICE)
    m.load_state_dict(torch.load("translator_dim8.pt"))
    m.eval()
    code = torch.load("kannada_codebook.pt").to(DEVICE)

    (_, _), (Xte, yte) = load_tensors()
    Xte, yte = Xte.squeeze(1).to(DEVICE), yte.to(DEVICE)
    picks = torch.cat([torch.where(yte == c)[0][:4] for c in range(3)])
    imgs, labels = Xte[picks], yte[picks]          # 12 images

    with torch.no_grad():
        # ---- encoder stages -------------------------------------------
        x = imgs.unfold(1, P, P).unfold(2, P, P).reshape(-1, SEQ, P * P)
        ink = x.mean(-1)                            # 12,49 patch brightness
        E0 = m.patch(x)                             # 12,49,8
        E1 = E0 + m.enc_pos
        E2 = E1.clone()
        for blk in m.enc:
            E2 = blk(E2)
        allE = torch.cat([E0.reshape(-1, DIM), E1.reshape(-1, DIM),
                          E2.reshape(-1, DIM)])
        _, Vh, mu, var_enc = pca4(allE)
        stages = [rl(project(E.reshape(-1, DIM), Vh, mu)) for E in (E0, E1, E2)]
        meta = []
        for i in range(12):
            for p in range(SEQ):
                meta.append({"cls": int(labels[i]), "img": i, "r": p // 7,
                             "c": p % 7, "ink": round(float(ink[i, p]), 3)})

        # ---- per-head q/k/v (encoder block, input = E1) ---------------
        blk = m.enc[0]
        qin, kin = blk.norm_q(E1), blk.norm_kv(E1)
        heads = {}
        for name, lin, src in (("q", blk.q, qin), ("k", blk.k, kin),
                               ("v", blk.v, kin)):
            t = lin(src).view(12, SEQ, HEADS, 2)    # dh = 2
            heads[name] = [rl(t[:, :, h].reshape(-1, 2)) for h in range(HEADS)]

        # ---- decoder word space ---------------------------------------
        W = m.head.weight.detach()                  # 64,8 output directions
        Wp, VhW, muW, var_w = pca4(W)
        emb = m.tok.weight.detach()                 # 65,8 input embeddings
        Ep, _, _, var_e = pca4(emb)
        word_ink = [round(float(code[j].mean()), 3) for j in range(K)]

        # greedy generation trace, one input per class
        gen = {"tokens": [], "traj": []}
        for c in range(3):
            xi = Xte[torch.where(yte == c)[0][0]][None]
            mem = m.encode(xi)
            seq = torch.tensor([[BOS]], device=DEVICE)
            hs, toks = [], []
            for _ in range(SEQ):
                y = m.tok(seq) + m.dec_pos[:, :seq.shape[1]]
                for sa, ca in zip(m.dec_self, m.dec_cross):
                    y = sa(y, causal=True)
                    y = ca(y, kv=mem)
                h = y[0, -1]
                nxt = m.head(h).argmax().item()
                hs.append(h); toks.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE)], 1)
            gen["tokens"].append(toks)
            gen["traj"].append(rl(project(torch.stack(hs), VhW, muW)))

    # display-only english "alphabet": k-means over english test patches, for
    # intuition — the encoder never quantizes, it reads raw patches.
    from kannada_codebook import kmeans
    eng_flat = Xte.unfold(1, P, P).unfold(2, P, P).reshape(-1, P * P)
    sub = eng_flat[torch.randperm(len(eng_flat))[:200_000].to(DEVICE)]
    eng_code, _ = kmeans(sub.contiguous(), K)
    eng_code = eng_code[eng_code.sum(1).argsort()]

    data = {
        "imgs": [rl(im) for im in imgs.cpu()],
        "engCode": rl(eng_code.cpu()),
        "stageNames": ["patch embed", "+ position", "after attention"],
        "stages": stages, "meta": meta, "varEnc": [round(v, 3) for v in var_enc],
        "heads": heads,
        "W": rl(Wp), "emb": rl(Ep), "wordInk": word_ink,
        "varW": [round(v, 3) for v in var_w], "varEmb": [round(v, 3) for v in var_e],
        "gen": gen,
        "codebook": rl(code.cpu()),
        "clsNames": CLS_NAMES,
    }
    with open("translator_journey_data.js", "w") as f:
        f.write("window.TJ = " + json.dumps(data) + ";\n")
    print(f"encoder PCA var explained: {var_enc}")
    print(f"word-space PCA var explained: {var_w}")
    print("wrote translator_journey_data.js")


if __name__ == "__main__":
    main()
