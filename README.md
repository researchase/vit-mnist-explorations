# Transformers you can see: a ViT for MNIST, small enough to watch

This repository started as a Vision Transformer for MNIST and grew into a set of
**tiny, fully-inspectable transformers built for understanding** — models small
enough (hundreds to a few thousand parameters) that every weight, every attention
head, and every representational space can be plotted and watched rather than
merely measured.

It is organized around a single question: *if a ViT reads an image as a sentence
of patches, how far can you push that analogy?* Far enough, it turns out, to
translate between two handwritten scripts — that extension now lives in its own
repo, [visual-language-translation](https://github.com/researchase/visual-language-translation).

> **Adapted from** [shub-garg/Vision-Transformer-VIT-for-MNIST](https://github.com/shub-garg/Vision-Transformer-VIT-for-MNIST)
> by **Shubham Garg**, whose original ViT-for-MNIST implementation and notebook
> are the foundation this work builds on. The base `ViT` model and the original
> training notebook are his; the interpretability studio and the minimal-model
> ablations are extensions of that work.

---

## What's here

### 1. The minimal ViT — how small can a classifier be?
An attention-only ViT (MLPs removed) that classifies MNIST digits {0,1,2}.
Through a series of ablations it shrinks to **371 parameters at 98.8% accuracy**,
mapping out how patch size, width, and head count trade against each other.
- `min_model.py`, `patch4_experiment.py`, `dim_sweep.py`, `tiny_pool.py`, `dim4_fullres.py`

### 2. ViT Studio — the interactive explainer
A single-page, offline visualization of the whole classifier: every weight as a
cube, the CLS token's journey through the network in raw 4-D, a guided lesson,
and four zoom-in teaching modules as centre tabs:
- **the machine** — activations flowing through the real network
- **LayerNorm geometry** — normalization as projection onto a sphere
- **4D → 1D projection** — how a head reads out a scalar
- **q·kᵀ dot product** — attention scores as ‖k‖cosθ×‖q‖ → softmax

Open `index.html` (GitHub Pages landing page). Companion single-file apps:
`head_explorer.html`, `network_3d.html`, `vit_explainer.html`.

### 3. English → Kannada visual translator → **moved to its own repository**
The encoder–decoder extension — treating MNIST and Kannada-MNIST as two visual
languages and *translating* between them, with a decoder that paints the target
digit patch by patch — grew into its own project and now lives at
**[visual-language-translation](https://github.com/researchase/visual-language-translation)**.
It builds on the attention-only encoder developed here.

---

## Repository layout

```
*.html, *.js, *.json   the ViT Studio web bundle (GitHub Pages, served from root)
*.py                   training + experiment + data-export scripts
*.pt                   trained classifier checkpoints
Images/                README figures
results/               experiment output plots (sweeps, head maps, journeys)
logs/                  training run logs
Vision_Transformer_for_MNIST.ipynb   the original notebook
```

The web bundle is intentionally flat: `index.html` is the Pages entry point and
its explainer iframes and `*_data.js` inputs are resolved by sibling path, so the
studio and the Python scripts that generate its data live together at the root.

## Setup

```bash
# classifier / notebook (original)
jupyter notebook Vision_Transformer_for_MNIST.ipynb

# minimal-model ablations
python min_model.py
python patch4_experiment.py
```

The interactive HTML pages are self-contained — open them directly in a browser
(three.js / plotly are bundled locally, except the q·kᵀ module which uses a CDN).

## Data
- **MNIST** — handwritten Western digits, classes {0,1,2}

## License
MIT. Original copyright © 2024 Shubham Garg; extensions © 2026 Ashwin Venkat.
See [LICENSE](LICENSE).
