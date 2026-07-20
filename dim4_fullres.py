"""How good can full-res dim=4 get? Sweep heads and training length (attention-only, 1 block)."""
import numpy as np
import torch
import viz_journey as vj
from tiny_pool import load_tensors
from min_model import run

DEVICE = vj.DEVICE


def main():
    (Xtr, ytr), (Xte, yte) = load_tensors()
    Xtr, ytr, Xte, yte = Xtr.to(DEVICE), ytr.to(DEVICE), Xte.to(DEVICE), yte.to(DEVICE)
    print("full-res 28x28, patch_size=7, dim=4, attention-only, 1 block\n")
    print(f"{'heads':>5s} {'epochs':>6s} {'params':>7s} {'accuracy':>16s}")
    print("-" * 40)
    for heads in (1, 2, 4):
        for epochs in (6, 20, 50):
            p, mu, sd = run(Xtr, ytr, Xte, yte, 28, 7, 4, heads, epochs=epochs)
            print(f"{heads:>5d} {epochs:>6d} {p:>7,d}   {mu:>6.2f}% ± {sd:.2f}")
        print()


if __name__ == "__main__":
    main()
