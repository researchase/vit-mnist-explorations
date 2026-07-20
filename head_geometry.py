"""
Quantify the "each head adds one dimension" effect the eye sees in cls_4d_shapes.
For each journey stage we compute the covariance of the 450 CLS points and report:
 - eigenvalues (variance along each principal axis)
 - effective dimensionality = participation ratio (Σλ)² / Σλ²   (1 = a line, 4 = full 4-D ball)
 - total variance (cloud size)
Also prints the 4 output-projection column directions each head moves along, and their angles.
"""
import numpy as np
import torch
import viz_journey as vj
from min_model import build
from tiny_pool import load_tensors
from cls_4d import sample, trace
import os

DEVICE, HERE = vj.DEVICE, vj.HERE


def stats(X):                      # X: [N,4]
    C = np.cov(X.T)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    pr = (tot ** 2) / (ev @ ev) if (ev @ ev) > 1e-20 else float("nan")
    return ev, tot, pr


def main():
    m = build(28, 4, 4, 4)
    m.load_state_dict(torch.load(os.path.join(HERE, "vit_371.pt"), map_location=DEVICE)); m.eval()
    (_, _), (Xte, yte) = load_tensors()
    imgs, labels = sample(Xte, yte)
    names, arr = trace(m, imgs)        # [S,N,4]

    print(f"{'stage':36s} {'eigenvalues (var/axis)':30s} {'eff.dim':>7s} {'totalvar':>9s}")
    print("-" * 86)
    for s, nm in enumerate(names):
        ev, tot, pr = stats(arr[s])
        evs = " ".join(f"{e:5.2f}" for e in ev)
        print(f"{nm:36s} {evs:30s} {pr:>7.2f} {tot:>9.2f}")

    # the fixed direction each head pushes along = column of to_out weight
    W = m.transformer.layers[0].fn.to_out[0].weight.detach().cpu().numpy()  # [4,4]
    dirs = [W[:, h] / (np.linalg.norm(W[:, h]) + 1e-9) for h in range(4)]
    print("\nEach head moves the CLS token along ONE fixed direction (to_out column, dim_head=1):")
    for h in range(4):
        print(f"  head {h}: dir = [{', '.join(f'{v:+.2f}' for v in W[:,h])}]")
    print("\npairwise angles between head directions (deg):")
    for i in range(4):
        for j in range(i + 1, 4):
            ang = np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)))
            print(f"  head {i} · head {j}: {ang:5.1f}°")


if __name__ == "__main__":
    main()
