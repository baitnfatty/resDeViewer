"""Per-layer residual-stream decomposition capture for HF decoder-only LMs.

Captures, per decoder layer L, via forward hooks:
  - ``resid_pre[L]``  — the hidden state ENTERING block L (forward pre-hook on the
                        block). At L=0 this is the embedding stream.
  - ``attn_out[L]``   — the self-attention module's output: post-o_proj, summed
                        over heads, BEFORE the residual add (the pre-residual-add
                        delta). NOT per-head (see the per-head seam at the bottom).
  - ``mlp_out[L]``    — the MLP module's output, before the residual add.
  - ``resid_post[L]`` — the block's output (hidden state leaving block L).

Assumed block convention (Llama/Qwen/Mistral-style pre-norm):
    resid_post = resid_pre + attn_out + mlp_out
This identity is ASSUMED, not guaranteed by the hook API — ``selfcheck.py``
verifies it in fp32 before the tool is trusted on a given model/transformers
version. If the check fails, the module-output convention differs and this
decomposition is wrong for that model; do not use the tool on it.

Reconstruction helpers (per the tool's spec):
    resid_pre(L)  = resid_post(L-1), or the embedding stream at L=0
    resid_mid(L)  = resid_pre(L) + attn_out(L)
    resid_post(L) = captured directly

All states are RAW residual-stream vectors (no normalization applied) unless a
function says otherwise. The logit lens applies the model's REAL final norm
(e.g. ``model.model.norm``) followed by the real ``lm_head`` — not a per-layer
norm, and no tuned/learned lens.

PERFORMANCE NOTE (prototype): every capture does ``.detach().cpu()`` — fine for
single prompts. A scaled version should keep tensors on GPU and reduce there
(norms / cosines / lens on device), moving only the small results to CPU. Not
built yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Shown verbatim in the UI so the basis is never ambiguous (hard requirement:
# the tool is self-documenting). Update this string if conventions change.
BASIS_NOTES = """\
Basis & conventions shown by this tool:
- States are RAW residual-stream vectors (model's native basis, no normalization
  applied), except where a view is explicitly labeled otherwise.
- attn_out is the self-attention module output: post-o_proj, summed over heads,
  pre-residual-add. It is NOT a per-head decomposition (o_proj mixes heads).
- mlp_out is the MLP module output, pre-residual-add.
- resid_pre(L) enters block L; resid_mid(L) = resid_pre(L) + attn_out(L);
  resid_post(L) leaves block L. Identity resid_pre + attn_out + mlp_out =
  resid_post was verified in fp32 on this model (see selfcheck).
- Logit lens uses the model's REAL final norm + lm_head on the raw state
  (no per-layer norm, no tuned lens).
- This viewer is instrumentation, and its selfcheck certifies ONLY the
  decomposition identity. Any substantive finding produced with it (e.g.
  attention-vs-MLP attribution of a signal on some model) inherits the ACSL
  Rule 9 adversarial-review requirement at claim time — a passing selfcheck
  does not validate downstream claims.
"""

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

    @property
    def n_layers(self) -> int:
        return len(self.resid_post)

    def resid_mid(self, layer: int) -> torch.Tensor:
        """Reconstructed: resid_pre(L) + attn_out(L)."""
        return self.resid_pre[layer] + self.attn_out[layer]

    def reconstructed_resid_pre(self, layer: int) -> torch.Tensor:
        """Reconstructed per spec: prior layer's resid_post, or the embedding
        stream at L0 (falls back to the directly captured L0 input)."""
        if layer == 0:
            return self.embed_out if self.embed_out is not None else self.resid_pre[0]
        return self.resid_post[layer - 1]


class ResidualCapture:
    """Context manager that hooks a model and captures the decomposition.

    Usage:
        with ResidualCapture(model) as cap:
            model(**inputs)
        result = cap.result
    """

    def __init__(self, model):
        self.model = model
        layers, layer_path = find_decoder_layers(model)
        self.layers = layers
        self.result = CaptureResult(layer_path=layer_path)
        # Resolve submodule names on the first block; assume homogeneous stack.
        _, self.result.attn_name = _find_sub(layers[0], _ATTN_NAMES, "attention")
        _, self.result.mlp_name = _find_sub(layers[0], _MLP_NAMES, "mlp")
        self._handles: list = []

    @staticmethod
    def _grab(t: torch.Tensor) -> torch.Tensor:
        # Prototype: copy every capture to CPU. Scaled version should stay
        # on-GPU and reduce there (see module docstring).
        return t.detach()[0].cpu()

    def __enter__(self):
        r = self.result

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

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def _pad(lst: list, idx: int):
    while len(lst) <= idx:
        lst.append(None)


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


# ------------------------------------------- per-head attribution seam ------
# FUTURE EXTENSION — deliberately NOT implemented. attn_out above is post-
# o_proj, so head contributions are already mixed. Per-head attribution
# requires a forward hook on the attention module's o_proj capturing its INPUT
# ([B, T, n_heads * head_dim]), reshaping to [B, T, n_heads, head_dim], and
# multiplying each head's slice by the matching column block of o_proj.weight
# to get that head's additive write into the residual stream. Plug it in here;
# ResidualCapture.__enter__ is where the extra hook would be registered.

def register_per_head_hooks(*_args, **_kwargs):
    raise NotImplementedError(
        "Per-head attribution is a planned extension — hook o_proj's input and "
        "split by head-dim column blocks of o_proj.weight (see comment above).")
