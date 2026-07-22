"""English->Kannada visual translation as a full encoder-decoder transformer.

Encoder: the familiar attention-only ViT — 49 patch tokens (4x4 patches), no CLS.
Decoder: autoregressive over a discrete Kannada patch vocabulary (64 k-means
codes + BOS), causal self-attention + cross-attention into the encoder, painting
the output image one 4x4 patch at a time in raster order.

Pairing is unpaired-by-class: each English digit gets a *random* Kannada digit
of the same class as its target every epoch, so the decoder must learn the
distribution of Kannada handwriting, not one fixed target.

Run: translator.py [dim]   (default runs dim=4 and dim=16 back to back)
"""
import sys
import time
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiny_pool import load_tensors
import viz_journey as vj
from kannada_codebook import load as load_kannada, to_patches, P, K

DEVICE = vj.DEVICE
SEQ = 49
BOS = K  # token id 64


# ---------------------------------------------------------------- model
class Attn(nn.Module):
    """Pre-norm multi-head attention; self- or cross- depending on kv input."""
    def __init__(self, dim, heads, dim_head=None):
        super().__init__()
        # dim need not divide heads: project to heads*dim_head and back (as in
        # the original ViT code). Defaults to dim//heads when it divides.
        self.h = heads
        self.dh = dim_head if dim_head is not None else max(1, dim // heads)
        inner = self.h * self.dh
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.last_attn = None

    def forward(self, x, kv=None, causal=False):
        q_in, kv_in = self.norm_q(x), self.norm_kv(kv if kv is not None else x)
        B, Nq, _ = q_in.shape
        Nk = kv_in.shape[1]
        q = self.q(q_in).view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(kv_in).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(kv_in).view(B, Nk, self.h, self.dh).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / self.dh ** 0.5
        if causal:
            a = a.masked_fill(torch.triu(torch.ones(Nq, Nk, dtype=torch.bool,
                                                    device=x.device), 1), -1e9)
        a = a.softmax(-1)
        self.last_attn = a.detach()
        out = (a @ v).transpose(1, 2).reshape(B, Nq, -1)
        return x + self.o(out)


class Translator(nn.Module):
    def __init__(self, dim, heads, enc_depth=1, dec_depth=2):
        super().__init__()
        self.patch = nn.Linear(P * P, dim)
        self.enc_pos = nn.Parameter(torch.randn(1, SEQ, dim) * 0.02)
        self.enc = nn.ModuleList([Attn(dim, heads) for _ in range(enc_depth)])
        self.tok = nn.Embedding(K + 1, dim)
        self.dec_pos = nn.Parameter(torch.randn(1, SEQ, dim) * 0.02)
        self.dec_self = nn.ModuleList([Attn(dim, heads) for _ in range(dec_depth)])
        self.dec_cross = nn.ModuleList([Attn(dim, heads) for _ in range(dec_depth)])
        self.head = nn.Linear(dim, K)

    def encode(self, img):                          # img: B,28,28
        x = img.unfold(1, P, P).unfold(2, P, P).reshape(-1, SEQ, P * P)
        x = self.patch(x) + self.enc_pos
        for blk in self.enc:
            x = blk(x)
        return x

    def decode(self, tokens, mem):                  # tokens: B,T (BOS-prefixed)
        y = self.tok(tokens) + self.dec_pos[:, :tokens.shape[1]]
        for sa, ca in zip(self.dec_self, self.dec_cross):
            y = sa(y, causal=True)
            y = ca(y, kv=mem)
        return self.head(y)                         # B,T,K

    def forward(self, img, tgt):                    # teacher forcing
        mem = self.encode(img)
        inp = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], 1)
        return self.decode(inp, mem)

    @torch.no_grad()
    def generate(self, img, temperature=1.0, greedy=False):
        mem = self.encode(img)
        B = img.shape[0]
        seq = torch.full((B, 1), BOS, dtype=torch.long, device=img.device)
        xattn = []
        for _ in range(SEQ):
            logits = self.decode(seq, mem)[:, -1]
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial((logits / temperature).softmax(-1), 1)
            xattn.append(self.dec_cross[-1].last_attn[:, :, -1])  # B,heads,49
            seq = torch.cat([seq, nxt], 1)
        return seq[:, 1:], torch.stack(xattn, 1)    # B,49 tokens; B,49,heads,49


# ---------------------------------------------------------------- data
def prepare():
    (Xtr, ytr), (Xte, yte) = load_tensors()        # english mnist {0,1,2}
    Xtr, ytr = Xtr.squeeze(1).to(DEVICE), ytr.to(DEVICE)
    Xte, yte = Xte.squeeze(1).to(DEVICE), yte.to(DEVICE)
    code = torch.load("kannada_codebook.pt").to(DEVICE)
    Ktr_img, Ktr_y = load_kannada("train")
    tok = torch.cdist(to_patches(Ktr_img).to(DEVICE).reshape(-1, P * P),
                      code).argmin(1).view(-1, SEQ)  # N,49 token ids
    Ktr_y = torch.from_numpy(Ktr_y.astype(np.int64)).to(DEVICE)
    by_class = [torch.where(Ktr_y == c)[0] for c in range(3)]
    return (Xtr, ytr), (Xte, yte), tok, by_class, code


def pair_targets(y, tok, by_class):
    """Random same-class Kannada token sequence for each English label."""
    out = torch.empty(len(y), SEQ, dtype=torch.long, device=DEVICE)
    for c in range(3):
        m = y == c
        pick = by_class[c][torch.randint(len(by_class[c]), (int(m.sum()),),
                                         device=DEVICE)]
        out[m] = tok[pick]
    return out


def detokenize(tokens, code):
    rec = code[tokens].view(-1, 7, 7, P, P).permute(0, 1, 3, 2, 4)
    return rec.reshape(-1, 28, 28).cpu()


# ---------------------------------------------------------------- train / viz
def train(dim, heads, epochs=60, bs=256, lr=3e-3):
    (Xtr, ytr), (Xte, yte), tok, by_class, code = prepare()
    torch.manual_seed(0); np.random.seed(0)
    m = Translator(dim, heads).to(DEVICE)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n=== dim={dim} heads={heads}: {n:,} params ===")
    opt = optim.Adam(m.parameters(), lr=lr)
    N = len(Xtr)
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        tgt_all = pair_targets(ytr, tok, by_class)   # re-pair every epoch
        perm = torch.randperm(N, device=DEVICE)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = m(Xtr[idx], tgt_all[idx])
            loss = F.cross_entropy(logits.reshape(-1, K), tgt_all[idx].reshape(-1))
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 9 or ep == 0:
            m.eval()
            with torch.no_grad():
                tgt_te = pair_targets(yte, tok, by_class)
                te = F.cross_entropy(m(Xte, tgt_te).reshape(-1, K),
                                     tgt_te.reshape(-1)).item()
            print(f"ep {ep+1:3d}  train CE {tot/N:.4f}  test CE {te:.4f} "
                  f"(ppl {np.exp(te):.1f})  {time.time()-t0:.0f}s")
    torch.save(m.state_dict(), f"translator_dim{dim}.pt")
    return m, (Xte, yte), code


def sample_grid(m, Xte, yte, code, dim):
    """Rows: english input, then 3 sampled translations + 1 greedy."""
    fig, axes = plt.subplots(5, 12, figsize=(12, 5.6))
    picks = torch.cat([torch.where(yte == c)[0][:4] for c in range(3)])
    x = Xte[picks]
    outs = [x.cpu()]
    for s in range(3):
        torch.manual_seed(100 + s)
        t, _ = m.generate(x, temperature=1.0)
        outs.append(detokenize(t, code))
    tg, _ = m.generate(x, greedy=True)
    outs.append(detokenize(tg, code))
    labels = ["english in", "sample 1", "sample 2", "sample 3", "greedy"]
    for r in range(5):
        for c in range(12):
            axes[r][c].imshow(outs[r][c], cmap="gray", vmin=0, vmax=1)
            axes[r][c].axis("off")
        axes[r][0].set_ylabel(labels[r])
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
    plt.suptitle(f"english -> kannada translation, dim={dim} (3 temperature-1 "
                 f"samples + greedy per input)")
    plt.tight_layout()
    plt.savefig(f"translator_samples_dim{dim}.png", dpi=110)
    plt.close()


def xattn_grid(m, Xte, yte, code, dim):
    """For one input per class: where the decoder looks while painting."""
    steps = [3, 10, 17, 24, 31, 38, 45]
    fig, axes = plt.subplots(6, 1 + len(steps), figsize=(1.4 * (1 + len(steps)), 10))
    picks = [torch.where(yte == c)[0][0] for c in range(3)]
    for r, i in enumerate(picks):
        x = Xte[i:i + 1]
        torch.manual_seed(7)
        t, xa = m.generate(x, temperature=1.0)       # xa: 1,49,heads,49
        out = detokenize(t, code)[0]
        axes[2 * r][0].imshow(x[0].cpu(), cmap="gray")
        axes[2 * r][0].set_title("english in", fontsize=8)
        axes[2 * r + 1][0].imshow(out, cmap="gray")
        axes[2 * r + 1][0].set_title("kannada out", fontsize=8)
        for j, st in enumerate(steps):
            amap = xa[0, st].mean(0).view(7, 7).cpu()  # heads averaged
            axes[2 * r][j + 1].imshow(amap, cmap="viridis")
            axes[2 * r][j + 1].set_title(f"step {st}", fontsize=8)
            prog = torch.zeros(49); prog[:st + 1] = 1
            masked = (out.view(7, P, 7, P).permute(0, 2, 1, 3) *
                      prog.view(7, 7, 1, 1)).permute(0, 2, 1, 3).reshape(28, 28)
            axes[2 * r + 1][j + 1].imshow(masked, cmap="gray", vmin=0, vmax=1)
        for ax in axes[2 * r].tolist() + axes[2 * r + 1].tolist():
            ax.axis("off")
    plt.suptitle(f"cross-attention while painting (top: attn over english patches, "
                 f"bottom: canvas so far)  dim={dim}")
    plt.tight_layout()
    plt.savefig(f"translator_xattn_dim{dim}.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    dims = [int(sys.argv[1])] if len(sys.argv) > 1 else [4, 16]
    for dim in dims:
        m, (Xte, yte), code = train(dim, heads=4)
        m.eval()
        sample_grid(m, Xte, yte, code, dim)
        xattn_grid(m, Xte, yte, code, dim)
        print(f"saved translator_samples_dim{dim}.png, translator_xattn_dim{dim}.png")
