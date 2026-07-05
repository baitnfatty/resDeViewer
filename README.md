# resDeViewer — Residual-Stream Decomposition Viewer

A standalone mechanistic-analysis tool for inspecting how each layer of a
decoder-only transformer writes into the residual stream. Works on any
HuggingFace decoder LM (the model is a parameter, not a hardcoded checkpoint).

It is **instrumentation, not an evaluation harness**: it shows you the
decomposition; it makes no claims about what the numbers mean.

## What it does

For every decoder layer `L`, forward hooks capture:

| State | Source | Meaning |
|---|---|---|
| `resid_pre(L)` | block pre-hook (captured) | hidden state entering block L; the embedding stream at L0 |
| `attn_out(L)` | `self_attn` module output | attention's write: post-`o_proj`, summed over heads, **before** the residual add |
| `mlp_out(L)` | `mlp` module output | MLP's write, before the residual add |
| `resid_post(L)` | block output (captured) | hidden state leaving block L |

with the reconstructions `resid_mid(L) = resid_pre(L) + attn_out(L)` and
`resid_pre(L) = resid_post(L-1)` (validated, see below).

On a captured prompt it offers:

- **Per-component write norms** — ‖attn_out‖₂ and ‖mlp_out‖₂ per (layer, token),
  fp32-accumulated.
- **Write geometry** — relative norms ‖write‖/‖resid_pre‖ and write angles
  (cos vs resid_pre, cos attn-vs-mlp); erasure shows as negative cosine.
- **Cross-layer cosine** — cos between layers' `attn_out` or `mlp_out` writes at
  a chosen token position.
- **Logit lens** — any state (`resid_pre` / `resid_mid` / `resid_post`) decoded
  through the model's **real** final norm + `lm_head`, fp32 softmax. Top-1 grid
  over all (layer, token) cells, click-through top-k.
- **Per-head attribution** — one layer on demand: `o_proj`'s input split into
  head_dim column blocks × the matching `o_proj.weight` blocks = each head's
  additive write (bias, if any, is its own component so the sum closes
  exactly). Per-head norms + per-head lens. The head sum is asserted against
  the captured `attn_out` on every request.
- **Direct logit attribution (DLA)** — per-component contribution (embeddings,
  every attn/mlp write) to a chosen (position, target-token) logit via
  frozen-RMS linearization: exactly additive at the realized state, NOT a
  counterfactual. Gated on battery check F.
- **Attention patterns** — one layer's QK map [heads × T × T] from a dedicated
  eager-attention forward (sdpa exposes no weights); the prior implementation
  is restored and the additive identity re-asserted after every request,
  including error paths.
- **Interventions (ablation mode)** — re-capture with chosen writes
  zeroed/scaled (`layer:component:scale`). The un-intervened baseline is kept
  as a full capture of the same prompt, so every panel can diff against it;
  all views are loudly labeled counterfactual. Per-head attribution REFUSES
  when an intervention targets the same layer's attn output (the intervention
  bypasses `o_proj`, breaking the decomposition premise — verified, not
  precautionary).
- **Direction projection** — paste/upload any d_model vector at runtime;
  dot(state, v̂) and per-component write projections. Vectors arrive as request
  data, never as a repo path.
- **Session export** — every computed analysis + full provenance (model
  commit, dtype, device, prompt hash, interventions, gate-manifest pointer,
  code hashes) to a content-hash-named JSON under `captures/` (gitignored);
  optional raw `.npz`. Deterministic per (machine, dtype).
- **On-GPU mode (opt-in)** — captures stay on device and analyses reduce
  there, behind a runtime VRAM headroom guard that refuses (HTTP 507) rather
  than silently degrading. fp32 accumulation is enforced in both paths.

### Basis / conventions (also rendered verbatim in the UI)

- States are **raw residual-stream vectors** — no normalization applied.
- `attn_out` is **not** per-head (`o_proj` mixes heads before the tool sees it).
- The lens is the standard logit lens: the model's own final norm + unembedding,
  no per-layer norm, no tuned lens.

## The self-check gate (read this before trusting anything)

The decomposition assumes the pre-norm block convention
`resid_post = resid_pre + attn_out + mlp_out`. That assumption is **verified,
never trusted**: `selfcheck.py` loads the model in **forced float32** (fp16
reductions drift on this stack) and asserts the identity per layer at
tolerance 1e-4, on a fixed 4-prompt battery (short / 1430-token / chat-template
/ multilingual), plus six further checks:

- **B** chaining: `resid_pre(L) == resid_post(L-1)`;
- **C** L0 stream == embedding-module output;
- **D** lens on the final state == the model's own logits;
- **E** per-head sum == `attn_out` (mid layer, one extra forward);
- **F** DLA additivity: Σ frozen-RMS component contributions == final logits,
  against a tolerance **derived from the fp32 precision model**
  (ε·max|logit|·(K+√d)·C, inputs recorded in the manifest) — the measurement
  must fall UNDER the derived bound; it never sets the bar. Recorded SKIP
  (and panel refusal) if the model exposes no embedding module;
- **G** attention-pattern path: identity under the eager forward, map rows
  sum to 1, and the post-restore identity probe.

- **The gate is permanent and per-model.** A passing run writes a deterministic
  manifest to `manifests/selfcheck_<model-slug>.json` recording versions, model
  commit hash, code hashes, per-layer diffs, and `shortcuts_taken`. The UI
  server refuses to load any model whose manifest is missing, or was produced
  under a different torch/transformers version or different viewer code — it
  re-runs the battery first.
- **A failing identity check refuses the model.** No workaround exists in the
  code, by design: failure means the HF module-output convention differs on
  that model/version and the decomposition would be wrong.
- Two selfcheck runs produce **byte-identical manifests** (no timestamps, no
  stochastic ops); verify with `sha256sum`.

Validated so far: `Qwen/Qwen3-1.7B` @70d244cc on torch 2.9.1+rocm7.2.4 /
transformers 5.13.0 — identity checks A–D at **0.0**; E ≤ 8.4e-05 and
F ≤ 6.3e-05 (fp32 roundoff, ~20× under F's derived bound); G ≤ 7.8e-07.

## Usage

Both scripts run inside the pinned ROCm container (`rig:2.9.1`) with models
from the `hf_cache` volume, offline. On other machines, run the Python
entrypoints directly in any env with `torch` + `transformers` (+ `fastapi`,
`uvicorn` for the server).

```bash
# 1. Validate a model (permanent gate; rerun for every new model)
bash resid_viewer/run_selfcheck.sh                     # default Qwen3-1.7B
bash resid_viewer/run_selfcheck.sh --model <hf-id>     # any HF decoder

# 2. Start the UI
bash resid_viewer/run_server.sh                        # http://localhost:5001
bash resid_viewer/run_server.sh --stop
```

In the UI: **Load** a model (gated — runs the fp32 selfcheck first if no valid
manifest), **Run capture** on a prompt (optional chat-template wrap), then use
the norms / cosine / lens panels. The gate banner at the top always shows what
certified the loaded model.

### API (all JSON, port 5001)

| Endpoint | Does |
|---|---|
| `GET /api/status` | versions, gate state, basis notes, active attn impl, interventions |
| `GET /api/manifests` | certification inventory, validity re-checked live |
| `POST /api/load` | `{model, dtype: float32\|float16, revalidate}` — gated load |
| `POST /api/capture` | `{prompt, chat_template, interventions[], on_gpu}` → norms (+ baseline norms when intervened) |
| `POST /api/cosine` | `{component, token, which}` → L×L cosine matrix |
| `POST /api/lens_grid` / `lens_topk` | `{state, which, ...}` → lens decodes (current or baseline) |
| `POST /api/heads` | `{layer, token, top_k}` → per-head norms + lens; 409 on same-layer attn intervention |
| `POST /api/dla` | `{token, target}` → per-component contributions (+ baseline); 409 unless check F certified |
| `POST /api/geometry` | `{which}` → relative norms + write angles |
| `POST /api/project` | `{vector, state, which}` → direction projections; 400 on d_model mismatch |
| `POST /api/attn_pattern` | `{layer, head}` → QK map + eager/post-restore identity diffs |
| `POST /api/export` | `{include_raw}` → session JSON (+ optional .npz) under `captures/` |

## Using it *properly*

1. **Never skip the gate.** Results on a model without a passing manifest for
   the current versions/code are undefined.
2. **Findings inherit Rule 9.** The selfcheck certifies only the decomposition
   identity. Any substantive claim made *with* the viewer (e.g. "this signal is
   written by attention at L12, not the MLP") requires its own adversarial
   review before it is cited. Recorded in `BASIS_NOTES` and every manifest.
3. **Mind the dtype.** Analysis reductions (norms, softmax) accumulate in fp32
   internally, but a model *loaded* in float16 produces fp16 activations —
   fine for browsing; re-derive anything you intend to cite in float32.
4. **Read the basis panel** before interpreting a view: raw vs normalized,
   summed-heads attention, which norm the lens uses.

## Boundaries & current limits

- **No ACSL coupling.** Imports nothing from `acsl`; nothing imports it. Own
  container, own port (5001; the ACSL server is 5000). Both run simultaneously;
  they share only the GPU and the read-only model cache.
- **Performance.** Default path copies captures to CPU (fine interactively);
  opt-in on-GPU mode reduces on device behind the VRAM refusal guard. The
  lens grid still pushes full-vocab logits per layer sequentially.
- **Batch size 1 only.** Batched forward is not exercised by the selfcheck.
- **Counterfactual scope.** Interventions scale a module's OUTPUT — per-head
  attribution of an intervened layer's attn is therefore refused, and
  attention patterns fetched during an intervention are themselves
  counterfactual (labeled).

## Layout

```
capture.py      hooks, reconstruction, analyses, BASIS_NOTES, per-head seam
selfcheck.py    fp32 identity gate + battery + manifest writer (CLI & library)
server.py       FastAPI app enforcing the gate (port 5001)
frontend.html   single-file UI
run_selfcheck.sh / run_server.sh   container entrypoints
manifests/      committed selfcheck manifests (deterministic, one per model)
```
