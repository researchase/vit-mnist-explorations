# Painting in Translation: A 3,456-Parameter Encoder–Decoder Transformer that Translates Between Two Handwritten Scripts

**Ashwin Venkat** · with Claude (Anthropic) as research assistant
*July 2026 · working paper / educational study*

---

## Abstract

We treat handwritten digit recognition and generation as **machine translation
between two visual languages**. The source language is English-script MNIST: a
28×28 image read as a 49-token sentence of 4×4 pixel patches with a continuous
vocabulary. The target language is Kannada-script digits (Kannada-MNIST), given
a genuine finite vocabulary via a 64-entry k-means codebook over 4×4 patches.
A full encoder–decoder transformer in the style of Vaswani et al. (2017) —
attention-only, pre-norm, no MLPs — translates one into the other, painting the
Kannada digit autoregressively, one patch-word at a time. Crucially, source and
target images are **never paired**: each English digit receives a random
same-class Kannada digit as its target every epoch, so the decoder must learn
the *distribution* of Kannada handwriting rather than a fixed mapping.

At dim=8 with four 2-dimensional heads — **3,456 parameters, ~30 seconds of
training on one GPU** — the model translates all three classes {0,1,2}
correctly, greedy decoding yields a stable per-class prototype, and temperature
sampling yields diverse handwriting variants. Because the model is this small,
every representational space can be inspected directly. We report three
observations: (i) a **capacity ladder** on which greedy correctness, then
sample quality, emerge at measurable parameter counts; (ii) at fixed width and
near-identical perplexity, **head geometry decides *which* glyph the model
drops**, a miniature of an evaluation blind spot in language models; and
(iii) the decoder's input embedding arranges visual words by **usage, not
appearance** (distance-geometry correlation with pixel space: −0.07), while the
encoder's embedding mirrors appearance (0.82) — the model spontaneously
develops distributional semantics for an alphabet of ink strokes.

This is an educational study: every ingredient is known. Its contribution is
the combination — a complete, fully inspectable analogue of neural machine
translation in which "words," "sentences," and "meanings" are all visible as
pictures.

---

## 1. Introduction

The original transformer (Vaswani et al., 2017) was a machine-translation
model: an encoder reads a source sentence, a decoder writes a target sentence,
and cross-attention aligns the two. The Vision Transformer (Dosovitskiy et
al., 2020) kept only the encoder, replacing words with image patches and the
sentence with a classification token. This paper walks the remaining distance
in the opposite direction: if a ViT reads an image as a sentence, then
image-to-image conversion between two scripts should be *translation*, with
everything that entails — vocabularies, word order, alignment, and a
distribution over valid outputs.

We choose the smallest setting in which all of this is real rather than
metaphorical. MNIST digits {0,1,2} are the source language; Kannada-MNIST
digits (Prabhu, 2019) — a drop-in MNIST replacement of handwritten Kannada
numerals ೦, ೧, ೨ — are the target. Both are handwriting, so both sides have
the property that makes translation interesting: **many surface variants per
meaning**. There are thousands of ways to write "1" and thousands of ways to
write "೧," and no canonical pairing between them.

Two design decisions carry the analogy:

1. **A discrete target vocabulary.** The decoder does not regress pixels; it
   chooses, at each of 49 steps, one word from a finite alphabet of 64 patch
   shapes, via softmax. This makes vocabulary, perplexity, sampling, and
   temperature literal rather than borrowed vocabulary.
2. **Unpaired-by-class supervision.** Each training source image is paired
   with a *freshly sampled* same-class target image every epoch. There is no
   "the" correct translation, only a conditional distribution — as in natural
   language, where the average of all valid translations is not a valid
   translation.

The model is kept small enough — 3,456 parameters at the chosen operating
point — that its internal spaces can be plotted rather than probed. Width 8
with four heads means each attention head is exactly two-dimensional and can
be drawn on paper without dimensionality reduction.

### What this paper is not

None of the components are new. Discrete visual tokens with an autoregressive
transformer decoder are the mechanism of Image-GPT (Chen et al., 2020) and
DALL·E (Ramesh et al., 2021); k-means patch codebooks predate learned
quantization (VQ-VAE; van den Oord et al., 2017); tiny-transformer
interpretability has a rich literature (e.g., Elhage et al., 2021; Nanda et
al., 2023). The contribution is a *complete, minimal, fully visible instance*
of the translation framing, plus three empirical observations (§4–§6) that
fall out of being able to see everything at once.

---

## 2. Setup

### 2.1 Languages and data

**Source (English script).** MNIST filtered to classes {0,1,2}: ~18.6k train /
3.1k test images, 28×28 grayscale, cut into a 7×7 grid of 4×4 patches → 49
tokens per sentence, raster order. The vocabulary is continuous: any 16-pixel
patch is a possible word, embedded by a linear map.

**Target (Kannada script).** Kannada-MNIST filtered to {0,1,2}: 18k train / 3k
test. The vocabulary is made discrete: k-means (K=64) over all 4×4 patches of
the training split, codes sorted by ink. Reconstruction of held-out digits
from the codebook alone is legible (MSE 0.0097; 53/64 codes used), so the
64-word alphabet is sufficient to *write* Kannada digits. The alphabet
consists, as one would hope, of blanks, edges, and curve fragments — the
"letters" of a stroke-based script. One special word (BOS) starts every
sentence; no EOS is needed because sentences have fixed length 49.

### 2.2 Model

Attention-only (no MLPs), pre-norm, residual; dim *d*, *h* heads of dimension
*d/h*:

- **Encoder** — patch embedding (Linear 16→*d*), learned positions (49×*d*),
  one self-attention block. No CLS token: its pooling role is taken over by
  cross-attention. Output: 49 finished source tokens.
- **Decoder** — word embedding (65×*d*), learned positions, **two** layers of
  [causal self-attention → cross-attention into the encoder], and an output
  head (Linear *d*→64) whose rows are per-word direction vectors.

At the chosen operating point (*d*=8, *h*=4) the parameter budget is exact and
worth stating in full, because there is nowhere for anything to hide:

| component | params |
|---|---|
| patch embedding (16→8 + bias) | 136 |
| encoder positions (49×8) | 392 |
| encoder attention block | 288 |
| decoder word embedding (65×8) | 520 |
| decoder positions (49×8) | 392 |
| decoder blocks (2 × [self 288 + cross 288]) | 1,152 |
| output head (8→64 + bias) | 576 |
| **total** | **3,456** |

### 2.3 Training

Adam (lr 3·10⁻³), batch 256, 60 epochs, cross-entropy over all 49 positions
with teacher forcing. Targets are re-paired every epoch: an English "1" sees a
different Kannada "೧" each time. Training takes ~25–30 s per configuration on
an RTX 3090; the results below are from single seeds (seed 0) — a limitation,
see §7.

---

## 3. The model translates, and unpaired training does what it should

At *d*=16 (9,408 params) and *d*=8 (3,456 params), greedy decoding maps every
test input to a correct, legible glyph of the corresponding Kannada class, and
temperature-1 sampling produces *distinct handwriting variants* on each draw.
The greedy/sampled contrast is the point: greedy decoding collapses to a
per-class prototype (nearly identical across inputs of a class), while
sampling explores the learned conditional distribution. This is the
translation-not-regression lesson in visible form — an MSE decoder trained on
the same random pairing would average thousands of ೧s into a blur, and the
discrete decoder instead *commits*, patch by patch, to one coherent variant,
each committed patch conditioning the next through causal self-attention.

The written sentences themselves reward reading. Greedy output for an English
"0", as word indices over the 7×7 grid (word 0 = blank):

```
 0  0  0  0  0  0  0        · · · · · · ·
 0  0  0 60 50  0  0        · · · # # · ·
 0  0 36 51 46  0  0        · · # # # · ·
 0  0 46  0 46  0  0        · · # · # · ·
 0  0 46  0 46  0  0        · · # · # · ·
 0  0 47 57 51  0  0        · · # # # · ·
 0  0  0  0  0  0  0        · · · · · · ·
```

Three language-like statistics appear uninvited:

- **A function word.** Word 0 (blank) is 35 of 49 tokens — the "the"/space of
  the visual language, by far the most frequent word and the decoder's resting
  state.
- **Morpheme reuse.** Word 46 (a vertical stroke) serves as the left wall of
  ೦, the right wall of ೦, and both legs of ೧ — one sub-lexical unit composed
  into different words of the script.
- **A shared prefix with a late branch point.** The greedy sentences for
  English 0 and English 1 are nearly identical for rows 1–4 (the arch and two
  descending walls) and diverge only at row 5 (~step 37): ೦ closes its loop,
  ೧ leaves its legs open. Since the *canvas so far* is the same in both
  cases, the branch can only be forced by cross-attention consulting the
  source image — the model's equivalent of a translator glancing back at the
  text exactly when the languages diverge.

---

## 4. A capacity ladder: correctness first, distribution later

Shrinking the model reveals an ordered sequence of abilities:

| config | params | test CE (ppl) | greedy prototypes | temperature samples |
|---|---|---|---|---|
| d=4, 4h | 1,440 | 1.075 (2.9) | wrong / collapsed | noise |
| d=6, 4×1-D heads | 2,128 | 1.00 (2.7) | ೦ ✓, ೧ ✓, **೨ absent** | class-inconsistent |
| d=6, 2×3-D heads | 2,368 | 0.95 (2.6) | all three as skeletons; ೦ has an artifact | rough |
| d=6, 3×2-D heads | 2,368 | 0.98 (2.7) | balanced skeletons | rough |
| **d=8, 4h** | **3,456** | 0.87 (2.4) | **all clean** | right class, shaky strokes |
| d=16, 4h | 9,408 | 0.78 (2.2) | all clean | clean, diverse variants |

Two readings. First, the floor for *knowing what to paint* (greedy
correctness) sits between 1.4k and 3.5k parameters, and the floor for
*painting the distribution well* (sharp diverse samples) is higher — roughly
the 9k model. Task difficulty ordered the same architecture's requirements:
in earlier ablations on this codebase, *classifying* {0,1,2} needed only 371
parameters; painting the prototype needs ~3.5k; painting the distribution
wants ~9k.

Second, degradation is not uniform. As capacity shrinks the model loses the
hardest glyph first — ೨, with its compositional structure (curled top, then a
long base stroke), degrades to a skeleton and then vanishes, while ೧ (a
single arch) survives in every configuration. A translation model below
capacity keeps the easy words and loses the hard ones: it speaks with an
accent before it falls silent.

---

## 5. Equal perplexity, different competence: head geometry at the floor

The three *d*=6 configurations differ only in how the width is split into
heads, and their test perplexities are within 0.1 of each other. Their
behavior is not:

- **4 heads × 1-D**: the cleanest ೦ and ೧ of the three — and no ೨ at all
  (greedy emits a stray tick). Many scalar probes seem well-suited to simple
  global properties (closed loop present?) and unable to coordinate the
  multi-stroke composition ೨ requires.
- **2 heads × 3-D**: all three glyphs present as skeletons, everything
  slurred, and a systematic artifact on ೦ (a hook intruding into the oval).
- **3 heads × 2-D**: the compromise; balanced but unremarkable.

At fixed parameter budget, **head geometry is a knob that redistributes which
words a model can say, nearly invisibly to the loss.** The scalar metric
genuinely cannot see the difference; only sampling reveals it. We believe this
is the study's most transferable observation, because it is a complete
miniature — inspectable end to end — of a known large-model phenomenon:
systems with indistinguishable benchmark scores can have meaningfully
different competence profiles, and the gap only appears under generation.

---

## 6. Four spaces, one width: the decoder invents distributional semantics

Width 8 invites a false conclusion: that "the" 8-dimensional space is shared
across the model. It is not, and because our words are pictures, this is
directly measurable. The 64 vocabulary words exist as 16-pixel patches, so
they can be placed into *four* different geometries: raw pixel space; the
encoder's patch embedding of those same patches; the decoder's input
embedding table; and the decoder's output-direction rows. Correlating the
64×63/2 pairwise word distances across spaces:

| | pixels | enc. embed | dec. input | dec. output |
|---|---|---|---|---|
| pixels | 1.00 | **0.82** | **−0.07** | 0.48 |
| encoder patch-embed | 0.82 | 1.00 | 0.13 | 0.38 |
| decoder input embedding | −0.07 | 0.13 | 1.00 | 0.22 |
| decoder output directions | 0.48 | 0.38 | 0.22 | 1.00 |

- The **encoder reads phonetically**: a linear map from pixels largely
  preserves pixel geometry (0.82). Words that look alike are neighbors.
- The **decoder's dictionary ignores appearance entirely** (−0.07 with
  pixels). Its words are arranged by context of use — which words precede
  and follow which — with no residue of what they look like. This is the
  word2vec phenomenon (Mikolov et al., 2013) reproduced from scratch, in 520
  parameters, over an alphabet of ink strokes: *"cat" and "dog" are neighbors
  because of how they are used, not how they are spelled.*
- Even the decoder's **two spaces disagree with each other** (0.22): "what it
  means to have just said word 46" and "what state makes me want to say word
  46 next" are different jobs and received different geometries. (Weight
  tying, standard in large LMs, would force these together; at this scale the
  model, left untied, chooses not to align them — a ready-made ablation.)

Nothing connects the encoder's and decoder's coordinate systems except the
learned cross-attention projections; any global rotation of one space,
compensated in those projections, leaves behavior unchanged. The
encoder–decoder symmetry (pictures→vectors in, vectors→picture-votes out) is
a symmetry of roles, not coordinates: two private languages, with
cross-attention as the only bilingual component.

---

## 7. Limitations

- **Single seeds.** The capacity ladder and head-geometry results are from
  one seed per configuration; the qualitative pattern (which glyphs survive)
  should be verified across seeds before being leaned on.
- **Three classes, one digit.** Sentences have trivial "syntax" — the
  interesting sequential structure (raster rhythm, shared prefixes) is
  spatial, not linguistic. The natural extension, multi-digit source images
  translated to multi-glyph target sequences, would make word order and
  alignment genuinely non-trivial.
- **k-means, not learned, quantization**; a fixed 4×4 grid; attention-only
  blocks. All chosen for inspectability over performance.
- **Evaluation is partly visual.** Perplexity is exact but, as §5 shows,
  insufficient; glyph-correctness judgments are by eye. A small Kannada
  classifier as a critic would make the ladder quantitative.
- The geometry correlations (§6) use Euclidean distances on 64 points in 8
  dimensions; other similarity measures (CKA, Procrustes) would strengthen
  the claim.

## 8. Artifacts

All code is single-file scripts in this repository; total compute for every
result in this paper is under five minutes on one consumer GPU.

- `kannada_codebook.py` — dataset download target, codebook construction,
  reconstruction check (`kannada_alphabet.png`, `kannada_codebook_recon.png`)
- `translator.py` — model, training, sample grids, cross-attention grids
  (`translator_samples_dim{4,6,6h3,6h4,8,16}.png`, `translator_xattn_*.png`)
- `export_translator_journey.py` + `translator_journey.html` — interactive
  visualization: the two vocabularies side by side; the encoder token journey
  (PCA 8→4 as x, y, z + color) with per-point patch inspection; each head's
  exact 2-D q/k/v spaces; and the decoder's output-direction space with the
  49-step painting trajectory, ink-coded, with a step slider — alongside the
  input-embedding dictionary view (§6).
- Checkpoints: `translator_dim{4,8,16}.pt`.

## References

- Vaswani et al., 2017. *Attention Is All You Need.*
- Dosovitskiy et al., 2020. *An Image is Worth 16×16 Words.*
- Prabhu, 2019. *Kannada-MNIST: A new handwritten digits dataset for the Kannada language.*
- van den Oord, Vinyals, Kavukcuoglu, 2017. *Neural Discrete Representation Learning (VQ-VAE).*
- Chen et al., 2020. *Generative Pretraining from Pixels (Image-GPT).*
- Ramesh et al., 2021. *Zero-Shot Text-to-Image Generation (DALL·E).*
- Mikolov et al., 2013. *Efficient Estimation of Word Representations in Vector Space.*
- Elhage et al., 2021. *A Mathematical Framework for Transformer Circuits.*
- Nanda et al., 2023. *Progress Measures for Grokking via Mechanistic Interpretability.*
