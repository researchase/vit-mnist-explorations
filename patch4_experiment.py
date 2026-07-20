"""
Patch-size experiment: break the image into 4x4 patches (7x7 grid -> 49 tokens + CLS)
vs the current 7x7 patches (4x4 grid -> 16 tokens + CLS).
Model: attention-only, 1 block, 4 heads; dim in {4, 8}; 50 epochs (tiny models need it), 3 seeds.
"""
import time
import numpy as np
import torch

import viz_journey as vj
from tiny_pool import load_tensors
from min_model import build, run, n_params

DEVICE = vj.DEVICE


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    Xtr, ytr, Xte, yte = Xtr.to(DEVICE), ytr.to(DEVICE), Xte.to(DEVICE), yte.to(DEVICE)
    print("attention-only · 1 block · 4 heads · 50 epochs · 3 seeds\n")
    print(f"{'patch':>6s} {'grid':>6s} {'tokens':>7s} {'dim':>4s} {'params':>7s} "
          f"{'accuracy':>16s} {'time':>7s}")
    print("-" * 62)
    for dim in (4, 8):
        for ps in (7, 4):
            g = 28 // ps
            t0 = time.time()
            p, mu, sd = run(Xtr, ytr, Xte, yte, 28, ps, dim, 4, epochs=50)
            dt = time.time() - t0
            print(f"{ps:>4d}px {g}x{g:<4d} {g*g+1:>7d} {dim:>4d} {p:>7,d} "
                  f"  {mu:>6.2f}% ± {sd:.2f} {dt:>6.0f}s")
        print()


if __name__ == "__main__":
    main()
