/* Re-implementation of the attention-only ViT forward pass in plain JS.
 * Runs live in the browser so heads/blocks can be toggled interactively.
 * Exposes global HE.runForward(DATA, exampleIdx, off) -> {nodes, attn, probs, pred}.
 *
 * off: a Set of strings. "b<k>" disables whole block k (identity). "h<k>_<h>" disables head h
 * of block k (its contribution to every token is zeroed; the block's bias still applies).
 *
 * The CLS "journey" is the exact linear decomposition:
 *   CLS_out = CLS_in + bias + Σ_head contribution_head
 * so nodes list the cumulative point after +bias and after each head.
 */
(function (root) {
  function layernorm(x, w, b, eps) {
    const n = x.length;
    let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
    const inv = 1 / Math.sqrt(v + eps);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * w[i] + b[i];
    return out;
  }
  // y = W x  (+b), W is [out][in]
  function matvec(W, x, b) {
    const o = W.length, out = new Array(o);
    for (let i = 0; i < o; i++) {
      const row = W[i]; let s = b ? b[i] : 0;
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }
  function softmax(a) {
    let mx = -Infinity; for (const v of a) if (v > mx) mx = v;
    let s = 0; const e = a.map(v => { const t = Math.exp(v - mx); s += t; return t; });
    return e.map(v => v / s);
  }
  function proj(pca, v) {
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const comp = pca.comp[c]; let s = 0;
      for (let d = 0; d < v.length; d++) s += (v[d] - pca.mean[d]) * comp[d];
      out[c] = s;
    }
    return out;
  }

  // normalized 28x28 -> 16 patch tokens embedded to dim
  function embed(DATA, px) {
    const { patch, grid } = DATA.cfg;
    const norm = px.map(p => (p / 255 - 0.1307) / 0.3081);
    const tokens = [];
    // CLS first
    tokens.push(DATA.cls.slice());
    for (let gr = 0; gr < grid; gr++) {
      for (let gc = 0; gc < grid; gc++) {
        const pv = new Array(patch * patch);
        let k = 0;
        for (let r = 0; r < patch; r++) {
          for (let c = 0; c < patch; c++) {
            pv[k++] = norm[(gr * patch + r) * 28 + (gc * patch + c)];
          }
        }
        tokens.push(matvec(DATA.patchW, pv, DATA.patchB));
      }
    }
    // add positional
    for (let t = 0; t < tokens.length; t++)
      for (let d = 0; d < tokens[t].length; d++) tokens[t][d] += DATA.pos[t][d];
    return tokens; // [17][dim]
  }

  function runForward(DATA, exampleIdx, off) {
    off = off || new Set();
    const cfg = DATA.cfg, H = cfg.heads, DH = cfg.dimHead, dim = cfg.dim, T = 17;
    let x = embed(DATA, DATA.examples[exampleIdx].px);

    const nodes = [];
    const attn = {}; // key `${b}_${h}` -> [16] CLS->patch attention
    nodes.push({ type: "input", block: -1, p: proj(DATA.pca, x[0]),
                 label: "input · patch+CLS+pos" });

    for (let b = 0; b < cfg.depth; b++) {
      if (off.has("b" + b)) {
        nodes.push({ type: "blockoff", block: b, p: proj(DATA.pca, x[0]),
                     label: `block ${b + 1} · OFF (identity)` });
        continue;
      }
      const blk = DATA.blocks[b];
      // per-token normed
      const normed = x.map(tok => layernorm(tok, blk.lnW, blk.lnB, blk.eps));
      // qkv per token: qkvW [192][64] -> [q(64),k(64),v(64)]
      const q = [], k = [], v = [];
      for (let t = 0; t < T; t++) {
        const o = matvec(blk.qkvW, normed[t], null);
        q.push(o.slice(0, dim)); k.push(o.slice(dim, 2 * dim)); v.push(o.slice(2 * dim, 3 * dim));
      }
      // per head attention -> out[t][head*DH..]
      const outCat = Array.from({ length: T }, () => new Array(dim).fill(0));
      for (let h = 0; h < H; h++) {
        const base = h * DH;
        // attention rows for all queries (we need all tokens to advance x)
        for (let i = 0; i < T; i++) {
          const scores = new Array(T);
          for (let j = 0; j < T; j++) {
            let s = 0; for (let d = 0; d < DH; d++) s += q[i][base + d] * k[j][base + d];
            scores[j] = s * blk.scale;
          }
          const a = softmax(scores);
          if (i === 0) attn[b + "_" + h] = a.slice(1); // CLS -> 16 patches
          const disabled = off.has("h" + b + "_" + h);
          if (!disabled) {
            for (let d = 0; d < DH; d++) {
              let acc = 0; for (let j = 0; j < T; j++) acc += a[j] * v[j][base + d];
              outCat[i][base + d] = acc;
            }
          } // disabled -> leaves zeros
        }
      }
      // delta = to_out(outCat) for every token; CLS decomposed per head
      const delta = outCat.map(oc => matvec(blk.outW, oc, blk.outB));
      // CLS journey nodes
      let p = x[0].slice();
      for (let d = 0; d < dim; d++) p[d] += blk.outB[d]; // + bias
      nodes.push({ type: "bias", block: b, p: proj(DATA.pca, p), label: `block ${b + 1} · +bias` });
      for (let h = 0; h < H; h++) {
        const disabled = off.has("h" + b + "_" + h);
        if (!disabled) {
          // contribution of head h to CLS = outCat[0][slice] projected by outW columns
          const base = h * DH;
          for (let o = 0; o < dim; o++) {
            let s = 0; for (let d = 0; d < DH; d++) s += outCat[0][base + d] * DATA.blocks[b].outW[o][base + d];
            p[o] += s;
          }
        }
        nodes.push({ type: "head", block: b, head: h, disabled,
                     p: proj(DATA.pca, p),
                     label: `block ${b + 1} · head ${h}${disabled ? " (off)" : ""}` });
      }
      // advance residual stream
      for (let t = 0; t < T; t++) for (let d = 0; d < dim; d++) x[t][d] += delta[t][d];
    }

    // classifier head
    const feat = layernorm(x[0], DATA.head.lnW, DATA.head.lnB, DATA.head.eps);
    const logits = matvec(DATA.head.linW, feat, DATA.head.linB);
    const probs = softmax(logits);
    let pred = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[pred]) pred = i;
    return { nodes, attn, probs, pred };
  }

  root.HE = { runForward, proj };
})(typeof window !== "undefined" ? window : global);

if (typeof module !== "undefined") module.exports = (typeof window !== "undefined" ? window : global).HE;
