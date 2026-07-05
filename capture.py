"""Per-layer residual-stream decomposition capture for HF decoder-only LMs.

Captures, per decoder layer L, via forward hooks:
  - ``resid_pre[L]``  — the hidden state ENTERING block L (forward pre-hook on the
                        block). At L=0 this is the embedding stream.
  - ``attn_out[L]``   — the self-attention module's output: post-o_proj, summed
                        over heads, BEFORE the residual add (the pre-residual-add
                        delta). Per-head decomposition: :func:`capture_per_head`.
  - ``mlp_out[L]``    — the MLP module's output, before the residual add.
  - ``resid_post[L]`` — the block's output (hidden state leaving block L).

Assumed block convention (Llama/Qwen/Mistral-style pre-norm):
    resid_post = resid_pre + attn_out + mlp_out
This identity is ASSUMED, not guaranteed by the hook API — ``selfcheck.py``
verifies it in fp32 before the tool is trusted on a given model/transformers
version (permanent per-model gate). If the check fails, the decomposition is
wrong for that model; the tool refuses it.

Reconstruction helpers:
    resid_pre(L)  = resid_post(L-1), or the embedding stream at L=0
    resid_mid(L)  = resid_pre(L) + attn_out(L)

All states are RAW residual-stream vectors (no normalization applied) unless a
function says otherwise. The logit lens applies the model's REAL final norm
followed by the real ``lm_head`` — no per-layer norm, no tuned lens.

INTERVENTIONS (verified V5/V5c/V6): a capture may run with component writes
zeroed/scaled. PINNED INVARIANT: intervention hooks are registered BEFORE
capture hooks on the same module — the reversed order silently records the
PRE-intervention write (V5c measured both orderings). Every intervened capture
re-asserts the additive identity at exit.

ATTENTION PATTERNS (verified V2/V2d/V2e/V2f): sdpa exposes no attention
weights (a self_attn hook sees ``(Tensor, None)``), so pattern capture runs a
dedicated forward under eager attention, hooked at ONE layer (no all-layer
collection), then restores the prior implementation and re-asserts the
additive identity post-restore — restoration was measured bit-exact (V2e/V2f),
the assertion proves it held on each request.

PERFORMANCE: the default capture path copies every tensor to CPU (fine for
single prompts). ``keep_on_device=True`` keeps captures on the GPU and reduces
there (norms / cosines / geometry / lens on device, only results cross to
CPU) — guarded by a runtime VRAM headroom check (:func:`ensure_vram_headroom`)
that REFUSES with a message rather than silently degrading; sibling GPU load
is variable, so no static budget is assumed. ACCEPTANCE CRITERION carried into
the device path: all reductions accumulate in fp32 even over fp16-resident
captures (fp16 sum-of-squares overflows d_model reductions).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

# Shown verbatim in the UI so the basis is never ambiguous (hard requirement:
# the tool is self-documenting). Update this string if conventions change.
BASIS_NOTES = """\
Basis & conventions shown by this tool:
- States are RAW residual-stream vectors (model's native basis, no normalization
  applied), except where a view is explicitly labeled otherwise.
- attn_out is the self-attention module output: post-o_proj, summed over heads,
  pre-residual-add. The per-head panel decomposes it via o_proj's input split
  into head_dim column blocks (o_proj bias, if any, shown as its own component).
- mlp_out is the MLP module output, pre-residual-add.
- resid_pre(L) enters block L; resid_mid(L) = resid_pre(L) + attn_out(L);
  resid_post(L) leaves block L. Identity resid_pre + attn_out + mlp_out =
  resid_post was verified in fp32 on this model (see selfcheck).
- Logit lens uses the model's REAL final norm + lm_head on the raw state
  (no per-layer norm, no tuned lens).
- Direct logit attribution (DLA) uses a frozen-RMS linearization: the final
  norm's scale is frozen at the realized final state, which makes per-component
  contributions EXACTLY additive to the model's logits. It is an attribution
  identity at the realized state — NOT a counterfactual "what if only this
  component had written".
- Attention patterns come from a SEPARATE forward under eager attention (sdpa
  exposes no weights); eager and sdpa forwards differ at kernel-noise level
  (2.4e-07 max fp32 logit diff measured on a reference config; fp16 divergence
  is larger — dtype-scoped values live in the manifest). All other panels use
  the default forward. The additive identity is re-asserted after every
  pattern request's restoration.
- When interventions are active, every derived view is COUNTERFACTUAL (a
  modified forward) and the UI labels it as such; the additive identity is
  re-asserted on every intervened capture. Per-head attribution REFUSES when an
  intervention targets the same layer's attn output (the intervention bypasses
  o_proj, so the per-head decomposition premise is broken — verified V6c).
- This viewer is instrumentation, and its selfcheck certifies ONLY the
  decomposition identity. Any substantive finding produced with it (e.g.
  attention-vs-MLP attribution of a signal on some model) inherits the ACSL
  Rule 9 adversarial-review requirement at claim time — a passing selfcheck
  does not validate downstream claims.
"""

DEFAULT_TOL = 1e-4
FP32_EPS = 1.19e-07

# Known layer-container paths, tried in order; then a generic fallback.
_LAYER_PATHS = (
    "model.layers",          # Llama, Qwen, Mistral, Gemma, ...
    "transformer.h",         # GPT-2, GPT-J, Falcon (some)
    "gpt_neox.layers",       # GPT-NeoX / Pythia
    "model.decoder.layers",  # OPT
    "transformer.blocks",    # MPT
)
_FINAL_NORM_PATHS = (
    "model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
    "model.decoder.final_layer_norm", "transformer.norm_f",
)
_ATTN_NAMES = ("self_attn", "attn", "attention", "self_attention")
_MLP_NAMES = ("mlp", "feed_forward", "ffn")


class IdentityViolation(RuntimeError):
    """The additive identity failed where it was asserted at runtime."""


class VRAMRefusal(RuntimeError):
    """On-device capture would exceed available VRAM headroom — refused
    explicitly, never silently degraded or truncated."""


def ensure_vram_headroom(bytes_needed: int, safety: float = 1.3):
    """Runtime guard for on-device captures: checks ACTUAL free VRAM (sibling
    GPU load varies, so static budgets are unreliable) and raises VRAMRefusal
    on shortfall."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    if bytes_needed * safety > free:
        raise VRAMRefusal(
            f"on-device capture needs ~{bytes_needed/1e9:.2f} GB "
            f"(x{safety:g} safety) but only {free/1e9:.2f} GB of "
            f"{total/1e9:.2f} GB VRAM is free — refusing; use CPU mode or a "
            f"shorter prompt")


class DLAUnavailable(RuntimeError):
    """DLA requires the embedding-stream anchor; this capture has none."""


class PerHeadRefused(RuntimeError):
    """Per-head decomposition refused: an intervention targets this layer's
    attn output, which bypasses o_proj and breaks the decomposition premise
    (V6c: per-head sum reconstructs the PRE-intervention write)."""


def _resolve(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def find_decoder_layers(model) -> tuple[torch.nn.ModuleList, str]:
    """Return (ModuleList of decoder blocks, dotted path). Tries known paths,
    then falls back to the largest ModuleList of same-class modules."""
    for path in _LAYER_PATHS:
        layers = _resolve(model, path)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers, path
    best = None
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2
                and len({type(m) for m in mod}) == 1
                and (best is None or len(mod) > len(best[0]))):
            best = (mod, name)
    if best is None:
        raise ValueError(f"Could not locate decoder layers in {type(model).__name__}")
    return best


def find_final_norm(model) -> tuple[torch.nn.Module, str]:
    """Return (final pre-lm_head norm module, dotted path)."""
    for path in _FINAL_NORM_PATHS:
        norm = _resolve(model, path)
        if isinstance(norm, torch.nn.Module):
            return norm, path
    raise ValueError(
        f"Could not locate final norm in {type(model).__name__}; "
        f"pass it explicitly to logit_lens().")


def _find_sub(block, names: tuple[str, ...], kind: str):
    for n in names:
        if hasattr(block, n):
            return getattr(block, n), n
    raise ValueError(f"Could not find {kind} submodule on {type(block).__name__} "
                     f"(tried {names})")


def _first_tensor(out):
    """HF modules variously return Tensor or (Tensor, ...) — take the hidden state."""
    if isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Hook expected Tensor, got {type(out).__name__}")
    return out


# ------------------------------------------------------------ interventions --

@dataclass(frozen=True)
class Intervention:
    """Scale one component's write: scale=0.0 ablates it entirely."""
    layer: int
    component: str  # 'attn' | 'mlp'
    scale: float = 0.0

    def validate(self, n_layers: int):
        if self.component not in ("attn", "mlp"):
            raise ValueError(f"intervention component must be 'attn'|'mlp', got {self.component!r}")
        if not (0 <= self.layer < n_layers):
            raise ValueError(f"intervention layer {self.layer} out of range [0,{n_layers})")


def _scale_hook(scale: float):
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (out[0] * scale,) + tuple(out[1:])
        return out * scale
    return hook


def _register_interventions(layers, attn_name: str, mlp_name: str,
                            interventions) -> list:
    """Register intervention hooks. MUST be called before any capture hooks on
    the same modules (V5c invariant)."""
    handles = []
    for iv in interventions:
        iv.validate(len(layers))
        block = layers[iv.layer]
        mod = getattr(block, attn_name if iv.component == "attn" else mlp_name)
        handles.append(mod.register_forward_hook(_scale_hook(iv.scale)))
    return handles


# ------------------------------------------------------------------ capture --

@dataclass
class CaptureResult:
    """Per-layer captures for one forward pass. Tensors are [T, d_model] on CPU
    (batch dim squeezed; single-prompt tool)."""
    resid_pre: list[torch.Tensor] = field(default_factory=list)
    attn_out: list[torch.Tensor] = field(default_factory=list)
    mlp_out: list[torch.Tensor] = field(default_factory=list)
    resid_post: list[torch.Tensor] = field(default_factory=list)
    embed_out: torch.Tensor | None = None  # embedding-module output (L0 stream)
    layer_path: str = ""
    attn_name: str = ""
    mlp_name: str = ""
    interventions: tuple = ()

    @property
    def n_layers(self) -> int:
        return len(self.resid_post)

    @property
    def is_counterfactual(self) -> bool:
        return len(self.interventions) > 0

    def resid_mid(self, layer: int) -> torch.Tensor:
        """Reconstructed: resid_pre(L) + attn_out(L)."""
        return self.resid_pre[layer] + self.attn_out[layer]

    def reconstructed_resid_pre(self, layer: int) -> torch.Tensor:
        """Reconstructed per spec: prior layer's resid_post, or the embedding
        stream at L0 (falls back to the directly captured L0 input)."""
        if layer == 0:
            return self.embed_out if self.embed_out is not None else self.resid_pre[0]
        return self.resid_post[layer - 1]

    def identity_max_diff(self) -> float:
        """max over layers of |resid_pre + attn_out + mlp_out - resid_post|,
        accumulated in fp32."""
        return max(
            float((self.resid_pre[L].float() + self.attn_out[L].float()
                   + self.mlp_out[L].float() - self.resid_post[L].float())
                  .abs().max())
            for L in range(self.n_layers))


class ResidualCapture:
    """Context manager that hooks a model and captures the decomposition.

    Usage:
        with ResidualCapture(model, interventions=(...)) as cap:
            model(**inputs)
        result = cap.result

    With interventions, the additive identity is re-asserted at exit
    (IdentityViolation on failure) — the captured writes are the
    POST-intervention writes (V5c invariant: intervention hooks first).
    """

    def __init__(self, model, interventions=(), tol: float = DEFAULT_TOL,
                 keep_on_device: bool = False):
        self.model = model
        self.tol = tol
        self.keep_on_device = keep_on_device
        layers, layer_path = find_decoder_layers(model)
        self.layers = layers
        self.result = CaptureResult(layer_path=layer_path,
                                    interventions=tuple(interventions))
        # Resolve submodule names on the first block; assume homogeneous stack.
        _, self.result.attn_name = _find_sub(layers[0], _ATTN_NAMES, "attention")
        _, self.result.mlp_name = _find_sub(layers[0], _MLP_NAMES, "mlp")
        self._handles: list = []

    @staticmethod
    def estimate_bytes(n_layers: int, n_tokens: int, d_model: int,
                       dtype_size: int) -> int:
        """Resident bytes for one on-device capture (4 states per layer)."""
        return n_layers * 4 * n_tokens * d_model * dtype_size

    def _grab(self, t: torch.Tensor) -> torch.Tensor:
        # Default: copy to CPU. On-device mode keeps captures on the GPU;
        # analyses reduce there in fp32 and only results cross to CPU.
        t = t.detach()[0]
        return t if self.keep_on_device else t.cpu()

    def __enter__(self):
        r = self.result
        # PINNED INVARIANT (V5c): intervention hooks BEFORE capture hooks on
        # the same module, or captures silently record pre-intervention writes.
        self._handles += _register_interventions(
            self.layers, r.attn_name, r.mlp_name, r.interventions)

        def store(lst, idx):
            def hook(_mod, _inp, out):
                _pad(lst, idx)
                lst[idx] = self._grab(_first_tensor(out))
            return hook

        def store_pre(lst, idx):
            def hook(_mod, inp):
                _pad(lst, idx)
                lst[idx] = self._grab(_first_tensor(inp))
            return hook

        emb = self.model.get_input_embeddings()
        if emb is not None:
            def emb_hook(_m, _i, out):
                r.embed_out = self._grab(_first_tensor(out))
            self._handles.append(emb.register_forward_hook(emb_hook))

        for i, block in enumerate(self.layers):
            attn = getattr(block, r.attn_name)
            mlp = getattr(block, r.mlp_name)
            self._handles += [
                block.register_forward_pre_hook(store_pre(r.resid_pre, i)),
                attn.register_forward_hook(store(r.attn_out, i)),
                mlp.register_forward_hook(store(r.mlp_out, i)),
                block.register_forward_hook(store(r.resid_post, i)),
            ]
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if exc is None and self.result.is_counterfactual and self.result.n_layers:
            d = self.result.identity_max_diff()
            if d > self.tol:
                raise IdentityViolation(
                    f"intervened capture violates additive identity: "
                    f"max abs diff {d:.3e} > tol {self.tol:g}")
        return False


def _pad(lst: list, idx: int):
    while len(lst) <= idx:
        lst.append(None)


def _identity_probe(model, enc, tol: float = DEFAULT_TOL) -> float:
    """One plain capture forward; returns identity max diff (raises if > tol).
    Used to prove the model is healthy after attention-impl restoration."""
    with torch.no_grad(), ResidualCapture(model) as cap:
        model(**enc)
    d = cap.result.identity_max_diff()
    if d > tol:
        raise IdentityViolation(
            f"post-restore identity probe failed: {d:.3e} > tol {tol:g}")
    return d


# ---------------------------------------------------------------- analyses --

def component_write_norms(result: CaptureResult) -> dict[str, torch.Tensor]:
    """L2 norm of each component's write, per layer x token position.
    Returns {'attn_out': [L, T], 'mlp_out': [L, T]} (raw-residual basis).
    Norms accumulate in fp32 even for fp16 captures — fp16 reductions over
    d_model overflow (sum of squares can exceed fp16 max)."""
    return {
        "attn_out": torch.stack([t.float().norm(dim=-1) for t in result.attn_out]),
        "mlp_out": torch.stack([t.float().norm(dim=-1) for t in result.mlp_out]),
    }


def cross_layer_cosine(result: CaptureResult, component: str,
                       token: int = -1) -> torch.Tensor:
    """[L, L] cosine similarity between layers' writes of ``component``
    ('attn_out' or 'mlp_out') at one token position (default: last token).
    Raw-residual basis."""
    vecs = torch.stack([t[token] for t in getattr(result, component)]).float()
    vecs = F.normalize(vecs, dim=-1)
    return vecs @ vecs.T


def write_geometry(result: CaptureResult) -> dict[str, torch.Tensor]:
    """Per (layer, token), fp32: relative write norms ‖write‖/‖resid_pre‖ and
    write angles — cos(write, resid_pre) (reinforcing vs erasing) and
    cos(attn_out, mlp_out). Raw-residual basis."""
    eps = 1e-12
    rel, cos_r, cos_am = {}, {}, None
    pre = [t.float() for t in result.resid_pre]
    for comp in ("attn_out", "mlp_out"):
        w = [t.float() for t in getattr(result, comp)]
        rel[comp] = torch.stack([w[L].norm(dim=-1) / (pre[L].norm(dim=-1) + eps)
                                 for L in range(result.n_layers)])
        cos_r[comp] = torch.stack([
            F.cosine_similarity(w[L], pre[L], dim=-1) for L in range(result.n_layers)])
    cos_am = torch.stack([
        F.cosine_similarity(result.attn_out[L].float(), result.mlp_out[L].float(), dim=-1)
        for L in range(result.n_layers)])
    return {"rel_norm_attn": rel["attn_out"], "rel_norm_mlp": rel["mlp_out"],
            "cos_attn_resid": cos_r["attn_out"], "cos_mlp_resid": cos_r["mlp_out"],
            "cos_attn_mlp": cos_am}


def direction_projection(result: CaptureResult, vec: torch.Tensor,
                         state: str = "resid_post") -> dict[str, torch.Tensor]:
    """Projection (dot with the UNIT vector v̂) of a chosen state and of both
    component writes, per (layer, token), fp32. Raises ValueError on d_model
    mismatch."""
    d_model = result.resid_post[0].shape[-1]
    v = vec.detach().float().flatten().to(result.resid_post[0].device)
    if v.numel() != d_model:
        raise ValueError(f"direction has dim {v.numel()}, model d_model is {d_model}")
    v = v / (v.norm() + 1e-12)
    if state not in ("resid_pre", "resid_mid", "resid_post"):
        raise ValueError(f"unknown state '{state}'")
    def st(L):
        return result.resid_mid(L) if state == "resid_mid" else getattr(result, state)[L]
    return {
        "state_proj": torch.stack([st(L).float() @ v for L in range(result.n_layers)]),
        "attn_proj": torch.stack([t.float() @ v for t in result.attn_out]),
        "mlp_proj": torch.stack([t.float() @ v for t in result.mlp_out]),
    }


def logit_lens(state: torch.Tensor, model, final_norm: torch.nn.Module | None = None,
               top_k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a reconstructed state [T, d_model] (or [d_model]) through the
    model's REAL final norm + lm_head. Returns (top_k token ids, probs), fp32
    softmax. This is the standard logit lens: the same norm the model applies
    before its own unembedding — no per-layer norm, no tuned lens."""
    if final_norm is None:
        final_norm, _ = find_final_norm(model)
    lm_head = model.get_output_embeddings()
    p = next(lm_head.parameters())
    x = state.to(device=p.device, dtype=p.dtype)
    with torch.no_grad():
        logits = lm_head(final_norm(x)).float()
    probs = logits.softmax(dim=-1)
    top = probs.topk(top_k, dim=-1)
    return top.indices.cpu(), top.values.cpu()


# ------------------------------------------------- per-head attribution (U1) --

def capture_per_head(model, enc, layer: int, interventions=(),
                     tol: float = DEFAULT_TOL) -> dict:
    """Per-head write vectors for ONE layer, on demand (one extra forward).

    Mechanism (verified V3, 1.5e-08): hook o_proj's INPUT [T, H*dh], split into
    head_dim column blocks, multiply each by the matching column block of
    o_proj.weight — each product is that head's additive write into the
    residual stream. o_proj bias (absent on Qwen3) is returned as its own
    component so the sum identity closes exactly.

    Inherits ``interventions`` (V6a/b: sum identity holds under downstream and
    upstream interventions). REFUSES if an intervention targets THIS layer's
    attn output — the intervention bypasses o_proj, so per-head would silently
    decompose the PRE-intervention write (V6c: measured exactly that).

    Memory: H×T×d_model fp32 ≈ 187 MB at T=1430 on Qwen3-1.7B — per-layer
    on-demand only; an all-layers variant (~5 GB) is deliberately not offered.
    """
    for iv in interventions:
        if iv.layer == layer and iv.component == "attn":
            raise PerHeadRefused(
                f"intervention scales layer {layer}'s attn output downstream of "
                f"o_proj; per-head decomposition premise broken (V6c) — refuse")
    layers, _ = find_decoder_layers(model)
    block = layers[layer]
    attn, attn_name = _find_sub(block, _ATTN_NAMES, "attention")
    _, mlp_name = _find_sub(block, _MLP_NAMES, "mlp")
    o_proj = attn.o_proj  # per-head requires the o_proj convention
    n_heads = model.config.num_attention_heads
    dh = o_proj.in_features // n_heads

    cap: dict = {}
    handles = _register_interventions(layers, attn_name, mlp_name, interventions)
    handles += [
        o_proj.register_forward_pre_hook(
            lambda _m, i: cap.__setitem__("oin", _first_tensor(i).detach()[0].cpu())),
        attn.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("aout", _first_tensor(o).detach()[0].cpu())),
    ]
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    oin = cap["oin"].float()                     # [T, H*dh]
    W = o_proj.weight.detach().float().cpu()     # [d_model, H*dh]
    per_head = torch.stack([oin[:, h*dh:(h+1)*dh] @ W[:, h*dh:(h+1)*dh].T
                            for h in range(n_heads)])          # [H, T, d_model]
    total = per_head.sum(0)
    bias = None
    if o_proj.bias is not None:
        bias = o_proj.bias.detach().float().cpu()
        total = total + bias
    diff = float((total - cap["aout"].float()).abs().max())
    if diff > tol:
        raise IdentityViolation(
            f"per-head sum vs attn_out: {diff:.3e} > tol {tol:g} at layer {layer}")
    return {"per_head": per_head, "attn_out": cap["aout"], "bias": bias,
            "n_heads": n_heads, "head_dim": dh, "sum_max_abs_diff": diff,
            "counterfactual": len(tuple(interventions)) > 0}


# --------------------------------------------- direct logit attribution (U2) --

def dla_tolerance(max_abs_logit: float, n_components: int, d_model: int,
                  eps: float = FP32_EPS, C: float = 4.0) -> float:
    """First-principles fp32 roundoff bound for the DLA additivity check.
    Frozen-RMS DLA is EXACTLY additive in exact arithmetic; the only error is
    accumulation roundoff: eps * max|logit| * (K + sqrt(d_model)) * C.
    Validated V4c: measured error sits 16-20x under this bound and does not
    grow with K. Measurement must fall UNDER this bound or the check FAILS —
    measurement never sets the bar."""
    return eps * max_abs_logit * (n_components + math.sqrt(d_model)) * C


def _dla_setup(result: CaptureResult, model, token: int):
    if result.embed_out is None:
        raise DLAUnavailable(
            "no embedding-module output captured; DLA cannot anchor the "
            "telescoping sum at L0 — refusing rather than mislabeling resid_pre[0]")
    final_norm, _ = find_final_norm(model)
    lm_head = model.get_output_embeddings()
    x_final = result.resid_post[-1][token].float()
    eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))
    rms = torch.rsqrt(x_final.pow(2).mean() + eps)          # frozen scale
    # Follow the capture's device (CPU default, GPU in on-device mode).
    nw = final_norm.weight.detach().float().to(x_final.device)
    comps = [("embed", result.embed_out[token])]
    comps += [(f"attn_L{L}", result.attn_out[L][token]) for L in range(result.n_layers)]
    comps += [(f"mlp_L{L}", result.mlp_out[L][token]) for L in range(result.n_layers)]
    return final_norm, lm_head, rms, nw, comps


def dla_check(result: CaptureResult, model, token: int = -1) -> dict:
    """Check F: the full-vocab sum of frozen-RMS component contributions must
    equal lens(resid_post[-1]) within the precision-derived tolerance."""
    final_norm, lm_head, rms, nw, comps = _dla_setup(result, model, token)
    p = next(lm_head.parameters())
    with torch.no_grad():
        ref = lm_head(final_norm(result.resid_post[-1][token].to(p.device, p.dtype))
                      ).float().cpu()
        acc = torch.zeros_like(ref)
        for _name, w in comps:
            x = (w.float() * rms * nw).to(p.device, p.dtype)
            acc += lm_head(x).float().cpu()      # fp32 accumulation
        if getattr(lm_head, "bias", None) is not None:
            # lm_head bias enters ref once but each component matvec above
            # re-adds it; remove the extras so the telescoping stays exact.
            acc -= lm_head.bias.detach().float().cpu() * (len(comps) - 1)
    diff = float((acc - ref).abs().max())
    max_logit = float(ref.abs().max())
    d_model = result.resid_post[0].shape[-1]
    n_comp = len(comps) + (1 if getattr(lm_head, "bias", None) is not None else 0)
    tol = dla_tolerance(max_logit, n_comp, d_model)
    return {"max_abs_diff": diff, "tol": tol, "tol_inputs": {
                "eps": FP32_EPS, "max_abs_logit": max_logit,
                "n_components": n_comp, "d_model": d_model, "C": 4.0},
            "pass": diff <= tol, "token": token}


def dla_attribution(result: CaptureResult, model, token: int,
                    target_id: int) -> dict:
    """Panel: per-component scalar contribution to ONE target logit."""
    final_norm, lm_head, rms, nw, comps = _dla_setup(result, model, token)
    wu_row = lm_head.weight[target_id].detach().float().to(nw.device)
    contribs = [{"component": name,
                 "contribution": float((w.float() * rms * nw) @ wu_row)}
                for name, w in comps]
    total = sum(c["contribution"] for c in contribs)
    if getattr(lm_head, "bias", None) is not None:
        b = float(lm_head.bias[target_id])
        contribs.append({"component": "lm_head.bias", "contribution": b})
        total += b
    p = next(lm_head.parameters())
    with torch.no_grad():
        actual = float(lm_head(final_norm(
            result.resid_post[-1][token].to(p.device, p.dtype)))[target_id])
    return {"contributions": contribs, "sum": total, "actual_logit": actual,
            "sum_vs_actual_diff": abs(total - actual), "token": token,
            "target_id": target_id,
            "counterfactual": result.is_counterfactual}


# ------------------------------------------------ attention patterns (U3) ----

def capture_attention_pattern(model, enc, layer: int, interventions=(),
                              tol: float = DEFAULT_TOL) -> dict:
    """QK attention map [H, T, T] for ONE layer.

    Runs a dedicated forward under eager attention (sdpa exposes no weights —
    V2d: an sdpa self_attn hook sees ``(Tensor, None)``), hooked at only this
    layer (retained AND transient cost is one layer's map; model-level
    ``output_attentions`` would collect all layers, ~3.7 GB at T=1430). The
    prior implementation is restored afterwards INCLUDING on error paths, and
    a post-restore identity probe re-asserts the additive identity (V2e/V2f
    measured restoration bit-exact; the probe proves it held on THIS request).

    Also returns the additive-identity diff measured DURING the eager forward
    (used by selfcheck check G). Inherits ``interventions`` — the map is then
    counterfactual and labeled as such.
    """
    layers, _ = find_decoder_layers(model)
    block = layers[layer]
    attn, attn_name = _find_sub(block, _ATTN_NAMES, "attention")
    _, mlp_name = _find_sub(block, _MLP_NAMES, "mlp")
    prev_impl = getattr(model.config, "_attn_implementation", "sdpa")

    weights: dict = {}

    def weight_hook(_m, _i, out):
        if isinstance(out, tuple):
            w = next((x for x in out[1:]
                      if isinstance(x, torch.Tensor) and x.dim() == 4), None)
            if w is not None:
                weights["w"] = w.detach()[0].float().cpu()   # [H, T, T]

    model.set_attn_implementation("eager")
    pattern_exc = None
    eager_identity = None
    try:
        handles = _register_interventions(layers, attn_name, mlp_name, interventions)
        handles.append(attn.register_forward_hook(weight_hook))
        try:
            # Full capture during the SAME eager forward → identity-under-eager
            # for free (check G), without a second forward.
            with torch.no_grad(), ResidualCapture(model, interventions=interventions,
                                                  tol=tol) as cap:
                model(**enc)
            eager_identity = cap.result.identity_max_diff()
        finally:
            for h in handles:
                h.remove()
    except Exception as e:                       # noqa: BLE001 — re-raised below
        pattern_exc = e
    finally:
        model.set_attn_implementation(prev_impl)
    # Post-restore probe runs ALWAYS — a failed pattern request must not leave
    # the model in a dirty attention-implementation state undetected.
    post_restore_identity = _identity_probe(model, enc, tol)
    if pattern_exc is not None:
        raise pattern_exc
    if "w" not in weights:
        raise RuntimeError(
            "eager forward returned no attention-weights tensor from the "
            "hooked layer — transformers attention-module contract changed; "
            "refusing (no workaround)")
    w = weights["w"]
    row_sum_max_dev = float((w.sum(-1) - 1.0).abs().max())
    return {"weights": w, "layer": layer,
            "eager_identity_max_diff": eager_identity,
            "post_restore_identity_max_diff": post_restore_identity,
            "row_sum_max_dev": row_sum_max_dev,
            "restored_impl": prev_impl,
            "counterfactual": len(tuple(interventions)) > 0}


# ------------------------------------------------------ session export (U7) --

def build_session_export(meta: dict, analyses: dict) -> dict:
    """Deterministic export payload: no timestamps; same capture + analyses →
    identical bytes (scoped per machine and dtype — fp16 analyses are
    byte-stable only on the same hardware/kernels)."""
    return {"tool": "resid_viewer session export", "meta": meta,
            "analyses": analyses}


def write_session_export(dir_path, payload: dict) -> Path:
    """Write payload as canonical JSON named by its own content hash —
    deterministic and idempotent (re-exporting the same session rewrites the
    same file)."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, sort_keys=True, indent=1) + "\n"
    name = f"session_{hashlib.sha256(blob.encode()).hexdigest()[:16]}.json"
    path = dir_path / name
    path.write_text(blob)
    return path
