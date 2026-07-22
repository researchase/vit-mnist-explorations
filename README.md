# Transformers you can see: from a ViT classifier to a visual-language translator

This repository started as a Vision Transformer for MNIST and grew into a set of
**tiny, fully-inspectable transformers built for understanding** — models small
enough (hundreds to a few thousand parameters) that every weight, every attention
head, and every representational space can be plotted and watched rather than
merely measured.

It is organized around a single question: *if a ViT reads an image as a sentence
of patches, how far can you push that analogy?* Far enough, it turns out, to
translate between two handwritten scripts.

> **Adapted from** [shub-garg/Vision-Transformer-VIT-for-MNIST](https://github.com/shub-garg/Vision-Transformer-VIT-for-MNIST)
> by **Shubham Garg**, whose original ViT-for-MNIST implementation and notebook
> are the foundation this work builds on. The base `ViT` model and the original
> training notebook are his; the interpretability studio, the minimal-model
> ablations, and the visual-language translator are extensions of that work.

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

### 3. English → Kannada visual translator — a transformer that *paints*
The full encoder–decoder transformer, restored to its original translation form.
It treats **MNIST digits and Kannada-MNIST digits as two visual languages** and
translates between them:
- the **encoder** reads an English digit as 49 patch tokens (no CLS);
- the **decoder** writes the Kannada digit autoregressively, choosing one word
  per step from a 64-entry visual vocabulary (a k-means codebook of 4×4 patches)
  and painting the output patch by patch, aligned to the source via
  cross-attention;
- source and target are **never paired** — each English digit gets a random
  same-class Kannada digit as its target every epoch, so the decoder learns the
  *distribution* of handwriting, not a fixed mapping.

At **3,456 parameters** it translates all three classes; greedy decoding gives a
per-class prototype, temperature sampling gives varied handwriting.
- `kannada_codebook.py` — build the visual alphabet + reconstruction check
- `translator.py` — model, training, sample & cross-attention grids
- `export_translator_journey.py` + `translator_journey.html` — interactive 3D
  view: the two vocabularies side by side, the encoder token journey, each
  head's raw 2-D q/k/v space, and the decoder's word-space with a step-slider
  that paints the digit as you scrub
- `PAPER.md` — a short write-up of three findings (a capacity ladder; how head
  geometry decides *which* glyph a model drops at equal perplexity; and how the
  decoder invents usage-based word embeddings while the encoder stays
  appearance-based)

---

## Setup

```bash
# classifier / notebook (original)
jupyter notebook Vision_Transformer_for_MNIST.ipynb

# translator: fetch Kannada-MNIST, build the codebook, then train
python kannada_codebook.py        # downloads to the data dir, builds alphabet
python translator.py              # trains dim=4 and dim=16 variants
python export_translator_journey.py   # data for the interactive viz
```

The interactive HTML pages are self-contained — open them directly in a browser
(three.js / plotly are bundled locally, except the q·kᵀ module which uses a CDN).

## Data
- **MNIST** — handwritten Western digits (the source language)
- **[Kannada-MNIST](https://github.com/vinayprabhu/Kannada_MNIST)** (Prabhu, 2019)
  — handwritten Kannada digits ೦–೯, a drop-in MNIST replacement (the target language)

## License
MIT. Original copyright © 2024 Shubham Garg; extensions © 2026 Ashwin Venkat.
See [LICENSE](LICENSE).
