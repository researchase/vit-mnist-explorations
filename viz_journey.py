"""
ViT representation-journey visualization on MNIST digits {0, 1, 2}.

Idea: an image's CLS-token representation (64-dim) is captured at every stage of the
encoder (input -> each attention add -> each MLP add). All stages live in the same
64-dim residual space, so a SINGLE fixed PCA basis projects every stage to 3D and we can
literally watch each image's point migrate from a mixed blob into 3 class clusters.

Reuses the ViT/Transformer/Attention/FeedForward classes from train_gpu.py unchanged.
"""
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from einops import repeat
import torchvision
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Reuse the model definitions from the training script (same folder)
from train_gpu import ViT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/mnt/nvme1tb/data/mnist"
CLASSES = [0, 1, 2]
CLASS_COLORS = ["#e6194B", "#3cb44b", "#4363d8"]  # 0=red, 1=green, 2=blue
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------------------------------------------
# Data: MNIST restricted to {0,1,2}
# --------------------------------------------------------------------------------------
def make_loaders():
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=tfm)
    test = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=tfm)

    def keep(ds):
        mask = torch.isin(ds.targets, torch.tensor(CLASSES))
        ds.data = ds.data[mask]
        ds.targets = ds.targets[mask]  # already 0/1/2, no remap needed
        return ds

    train, test = keep(train), keep(test)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
    return train_loader, test_loader


# --------------------------------------------------------------------------------------
# Model factory (patch_size=7 -> 4x4=16 patches, depth=4, heads=4)
# --------------------------------------------------------------------------------------
def make_model():
    return ViT(image_size=28, patch_size=7, num_classes=len(CLASSES), channels=1,
               dim=64, depth=4, heads=4, dim_head=16, mlp_dim=128).to(DEVICE)


def train_model(model, train_loader, test_loader, epochs=5):
    opt = optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            loss = F.nll_loss(F.log_softmax(model(data), dim=1), target)
            loss.backward()
            opt.step()
        acc = evaluate(model, test_loader)
        print(f"  epoch {ep}: test acc {acc*100:.2f}%")
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        pred = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.numel()
    return correct / total


# --------------------------------------------------------------------------------------
# The journey: replay the forward pass and stash the CLS token after every residual add.
# Returns stage_names + array [n_stages, N, dim] of CLS vectors.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def trace_cls(model, imgs):
    model.eval()
    imgs = imgs.to(DEVICE)
    x = model.to_patch_embedding(imgs)
    b, n, _ = x.shape
    cls = repeat(model.cls_token, "() n d -> b n d", b=b)
    x = torch.cat((cls, x), dim=1)
    x = x + model.pos_embedding[:, : (n + 1)]

    stages, names = [], []
    stages.append(x[:, 0].clone()); names.append("S0: input\n(patch+CLS+pos)")
    for i, (attn, ff) in enumerate(model.transformer.layers, start=1):
        x = attn(x) + x
        stages.append(x[:, 0].clone()); names.append(f"block {i}\nafter attention")
        x = ff(x) + x
        stages.append(x[:, 0].clone()); names.append(f"block {i}\nafter MLP")

    arr = torch.stack(stages, dim=0).cpu().numpy()  # [n_stages, N, dim]
    return names, arr


# --------------------------------------------------------------------------------------
# Collect a balanced sample of test images (per_class each of 0/1/2)
# --------------------------------------------------------------------------------------
def sample_images(test_loader, per_class=200):
    buckets = {c: [] for c in CLASSES}
    for data, target in test_loader:
        for c in CLASSES:
            need = per_class - len(buckets[c])
            if need > 0:
                sel = data[target == c][:need]
                buckets[c].append(sel)
        if all(len(torch.cat(v)) >= per_class for v in buckets.values() if v):
            if all(sum(t.shape[0] for t in buckets[c]) >= per_class for c in CLASSES):
                break
    imgs = torch.cat([torch.cat(buckets[c])[:per_class] for c in CLASSES], dim=0)
    labels = np.concatenate([np.full(per_class, c) for c in CLASSES])
    return imgs, labels


# --------------------------------------------------------------------------------------
# Fit ONE PCA basis on all stages pooled, project every stage through it.
# --------------------------------------------------------------------------------------
def project_all(arr):
    n_stages, N, dim = arr.shape
    pooled = arr.reshape(-1, dim)
    pca = PCA(n_components=3, random_state=0).fit(pooled)
    proj = pca.transform(pooled).reshape(n_stages, N, 3)
    return proj, pca


def stage_silhouettes(arr, labels):
    scores = []
    for s in range(arr.shape[0]):
        try:
            scores.append(silhouette_score(arr[s], labels))
        except Exception:
            scores.append(float("nan"))
    return scores


# --------------------------------------------------------------------------------------
# Interactive Plotly HTML with a slider over stages
# --------------------------------------------------------------------------------------
def build_html(proj, labels, names, sil, out_path, title):
    # consistent axis ranges across stages
    lo = proj.min(axis=(0, 1)); hi = proj.max(axis=(0, 1))
    pad = (hi - lo) * 0.05
    rng = [(lo[i] - pad[i], hi[i] + pad[i]) for i in range(3)]

    frames, slider_steps = [], []
    for s, name in enumerate(names):
        data = []
        for ci, c in enumerate(CLASSES):
            m = labels == c
            data.append(go.Scatter3d(
                x=proj[s, m, 0], y=proj[s, m, 1], z=proj[s, m, 2],
                mode="markers", name=f"digit {c}",
                marker=dict(size=3, color=CLASS_COLORS[ci], opacity=0.75),
            ))
        frames.append(go.Frame(data=data, name=str(s)))
        clean = name.replace("\n", " ")
        slider_steps.append(dict(method="animate", label=clean,
                                 args=[[str(s)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=f"{title}<br><sub>slide through stages — silhouette shown per stage</sub>",
        scene=dict(
            xaxis=dict(title="PC1", range=rng[0]),
            yaxis=dict(title="PC2", range=rng[1]),
            zaxis=dict(title="PC3", range=rng[2]),
        ),
        sliders=[dict(active=0, steps=slider_steps,
                      currentvalue=dict(prefix="Stage: "))],
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=0.0,
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=700, redraw=True),
                                                         fromcurrent=True,
                                                         transition=dict(duration=300))])])],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    # annotate silhouette per frame via title update in each frame
    for s, fr in enumerate(frames):
        fr.layout = go.Layout(title=f"{title} — {names[s].replace(chr(10),' ')}"
                                    f"  (silhouette={sil[s]:.3f})")
    fig.write_html(out_path, include_plotlyjs="cdn", auto_open=False)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
# Static small-multiples PNG (matplotlib 3D), one panel per stage
# --------------------------------------------------------------------------------------
def build_grid_png(proj, labels, names, sil, out_path, suptitle):
    n = len(names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 4.2, rows * 4.0))
    for s in range(n):
        ax = fig.add_subplot(rows, cols, s + 1, projection="3d")
        for ci, c in enumerate(CLASSES):
            m = labels == c
            ax.scatter(proj[s, m, 0], proj[s, m, 1], proj[s, m, 2],
                       s=6, c=CLASS_COLORS[ci], alpha=0.6, label=f"{c}")
        ax.set_title(f"{names[s].replace(chr(10),' ')}\nsil={sil[s]:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if s == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------------------
def run_for_model(model, tag, imgs, labels, out_prefix, title):
    names, arr = trace_cls(model, imgs)
    proj, _ = project_all(arr)
    sil = stage_silhouettes(arr, labels)
    print(f"[{tag}] silhouette by stage:")
    for nm, sv in zip(names, sil):
        print(f"    {nm.replace(chr(10),' '):28s}  {sv:.3f}")
    build_html(proj, labels, names, sil, f"{out_prefix}.html", title)
    build_grid_png(proj, labels, names, sil, f"{out_prefix}_grid.png", title)
    return sil


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = make_loaders()

    # Balanced test sample used for BOTH trained & untrained (same points -> fair comparison)
    imgs, labels = sample_images(test_loader, per_class=200)
    print(f"Sample: {len(labels)} images ({', '.join(str(c) for c in CLASSES)})")

    # 1) Untrained baseline (random init) -- captured BEFORE training
    print("\n== Untrained (random init) baseline ==")
    untrained = make_model()
    run_for_model(untrained, "untrained", imgs, labels,
                  os.path.join(HERE, "pca_journey_untrained"),
                  "UNTRAINED ViT — representations never separate")

    # 2) Train the same architecture, then trace
    print("\n== Training 3-class ViT ==")
    trained = make_model()
    train_model(trained, train_loader, test_loader, epochs=5)
    torch.save(trained.state_dict(), os.path.join(HERE, "vit_mnist_3class.pt"))
    print("\n== Trained model journey ==")
    run_for_model(trained, "trained", imgs, labels,
                  os.path.join(HERE, "pca_journey"),
                  "TRAINED ViT — CLS token separates into 3 clusters")

    print("\nDone. Open pca_journey.html and pca_journey_untrained.html in a browser.")


if __name__ == "__main__":
    main()
