"""fp32 self-validation for resid_viewer's decomposition assumption. This is a
PERMANENT GATE, not a one-time rite: it certifies one (model x torch x
transformers x code) combination, and must be rerun for any new model before
the viewer's decomposition is trusted on it. The UI server enforces this.

    bash resid_viewer/run_selfcheck.sh          # one command, from repo root
    # or inside the container:
    python resid_viewer/selfcheck.py [--model HF_ID] [--prompt ...] [--device ...]

Primary identity (per decoder layer L, tol 1e-4 max abs diff in fp32):

    resid_pre(L) + attn_out(L) + mlp_out(L) == resid_post(L)   (captured directly)

The model is loaded in float32 for this check REGARDLESS of what it will later
run in — fp16 reductions drift on this stack (see ACSL CLAUDE.md numerics
notes; that fact is stack-level, not ACSL-specific). If the identity FAILS, the
HF module-output convention on this transformers version differs from the
assumed pre-residual-add delta: the correct response is to report the diffs and
STOP — no workarounds, no loosened tolerance.

Secondary checks (validate the reconstruction + lens paths the tool offers):
  B. chaining:   resid_pre(L) == resid_post(L-1)          (reconstruction rule)
  C. L0 stream:  resid_pre(0) == embedding-module output   (models that scale
                 embeddings after embed_tokens, e.g. Gemma, will fail this —
                 that is a real finding about using "embeddings" at L0 there)
  D. lens anchor: logit_lens(resid_post[last]) == the model's own logits

Prompt battery: by default all checks run on a FIXED battery of 4 prompts
(short raw / ~long raw / chat-template / multilingual+symbols) defined below —
hardcoded constants, not sampled, so there is nothing to seed. The identity
under test is structural (a property of the block's forward code path), so the
battery buys code-path coverage (sequence length, template tokens, tokenizer
regimes), not statistics. `--prompt` replaces the battery with that single
prompt (recorded in the manifest as such).

Determinism: the script contains NO stochastic operation (eval mode, no
dropout, no sampling, fixed prompts). `torch.manual_seed(0)` is set defensively
and recorded in the manifest. The manifest is written WITHOUT timestamps so two
runs must produce byte-identical files (CLAUDE-level rule: two runs = identical
outputs); verify with `sha256sum` after re-running.

Manifest: results + environment + `shortcuts_taken` are written to
resid_viewer/manifests/selfcheck_<model-slug>.json — every number printed here
is also in that file on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resid_viewer.capture import (  # noqa: E402
    DLAUnavailable, IdentityViolation, ResidualCapture,
    capture_attention_pattern, capture_per_head, dla_check, find_final_norm)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
SEED = 0

RULE9_INHERITANCE = (
    "This manifest certifies instrumentation only (the decomposition identity "
    "on the model above). Any substantive finding produced with this viewer — "
    "e.g. attention-vs-MLP attribution of a signal on some model — inherits "
    "the ACSL Rule 9 adversarial-review requirement at claim time; a passing "
    "selfcheck does not validate downstream claims.")

_SENTENCES = (
    "The residual stream of a transformer accumulates writes from attention and MLP blocks at every layer.",
    "Each decoder block reads its input through a pre-norm, computes a delta, and adds it back to the stream.",
    "Logit lens decodes an intermediate state through the model's own final norm and unembedding matrix.",
    "Attention output here means the post-o_proj sum over heads, before the residual addition.",
    "Sequence length, template tokens, and tokenizer regime are varied across this battery deliberately.",
)

# Fixed battery — constants, not samples. Names are stable keys in the manifest.
PROMPT_BATTERY: tuple[tuple[str, str, bool], ...] = (
    ("short_raw", _SENTENCES[0], False),
    ("long_raw", " ".join(f"Point {i}: {_SENTENCES[i % len(_SENTENCES)]}"
                          for i in range(60)), False),
    ("chat_template", "Explain what a residual stream is in one paragraph.", True),
    ("multilingual_symbols",
     "Le flux résiduel накапливает 写入 from attention — αβγ, 🧮, and § symbols too.",
     False),
)


def manifest_path_for(model_id: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-").lower()
    return MANIFEST_DIR / f"selfcheck_{slug}.json"


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_sha256() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {f: _file_sha256(here / f) for f in ("capture.py", "selfcheck.py")}


def load_valid_manifest(model_id: str) -> tuple[dict | None, str]:
    """Return (manifest, "ok") if a manifest on disk still certifies model_id
    under the CURRENT torch/transformers versions and CURRENT code hashes;
    else (None, reason). Used by the UI server as the permanent gate."""
    import transformers
    path = manifest_path_for(model_id)
    if not path.exists():
        return None, f"no manifest at {path.name}"
    m = json.loads(path.read_text())
    if m.get("model") != model_id:
        return None, "manifest is for a different model"
    if m.get("torch_version") != torch.__version__:
        return None, f"torch changed ({m.get('torch_version')} -> {torch.__version__})"
    if m.get("transformers_version") != transformers.__version__:
        return None, (f"transformers changed ({m.get('transformers_version')} -> "
                      f"{transformers.__version__})")
    if m.get("code_sha256") != code_sha256():
        return None, "viewer code changed since validation"
    if not m.get("battery_is_default"):
        return None, "manifest was a single-prompt run, not the default battery"
    if not all(rec.get("all_pass") for rec in m.get("checks", [])):
        return None, "manifest records a failing check"
    return m, "ok"


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def report_failure(layer: int, lhs: torch.Tensor, rhs: torch.Tensor, tol: float):
    d = (lhs.float() - rhs.float()).abs()
    n_bad = int((d > tol).sum().item())
    print(f"\n  FAIL detail — layer {layer}:")
    print(f"    max abs diff : {d.max().item():.6e}")
    print(f"    mean abs diff: {d.mean().item():.6e}")
    print(f"    elements > tol({tol:g}): {n_bad} / {d.numel()} "
          f"({100.0 * n_bad / d.numel():.2f}%)")
    tok, dim = divmod(int(d.argmax().item()), d.shape[-1])
    print(f"    worst element: token {tok}, dim {dim} "
          f"(lhs={lhs.float()[tok, dim].item():.6f}, rhs={rhs.float()[tok, dim].item():.6f})")


class IdentityCheckFailed(RuntimeError):
    """Primary identity violated — the decomposition is wrong for this model.
    No workaround exists by design."""


def _encode(tokenizer, prompt: str, use_chat_template: bool, device):
    if use_chat_template and tokenizer.chat_template:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             add_generation_prompt=True, tokenize=False)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def run_checks(model, tokenizer, name: str, prompt: str, use_chat_template: bool,
               device: str, tol: float) -> dict:
    """All four checks on one prompt. Returns a manifest record; raises
    IdentityCheckFailed on a primary-identity failure (report, no workaround)."""
    enc = _encode(tokenizer, prompt, use_chat_template, device)
    n_tok = enc["input_ids"].shape[1]
    with torch.no_grad(), ResidualCapture(model) as cap:
        out = model(**enc)
    r = cap.result

    # A. primary identity
    a_diffs = []
    for L in range(r.n_layers):
        lhs = r.resid_pre[L].float() + r.attn_out[L].float() + r.mlp_out[L].float()
        d = max_abs(lhs, r.resid_post[L])
        a_diffs.append(d)
        if d > tol:
            print(f"\n[{name}] [A] FAIL at layer {L} (diff {d:.6e} > tol {tol:g}). "
                  f"Module-output convention differs from the assumed "
                  f"pre-residual-add delta. STOPPING — no workaround applied.")
            report_failure(L, lhs, r.resid_post[L], tol)
            raise IdentityCheckFailed(
                f"layer {L}: max abs diff {d:.6e} > tol {tol:g} on prompt '{name}'")
    a_max = max(a_diffs)

    # B. chaining; C. L0 == embedding output; D. lens anchor
    b_max = max(max_abs(r.resid_pre[L], r.resid_post[L - 1])
                for L in range(1, r.n_layers))
    c_max = (max_abs(r.resid_pre[0], r.embed_out)
             if r.embed_out is not None else None)
    final_norm, norm_path = find_final_norm(model)
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        lens_logits = lm_head(final_norm(r.resid_post[-1].to(device))).float().cpu()
    d_max = max_abs(lens_logits, out.logits[0].float().cpu())

    # E. per-head sum == attn_out at the middle layer (one extra forward).
    #    capture_per_head raises IdentityViolation above tol — record either way.
    mid = r.n_layers // 2
    try:
        ph = capture_per_head(model, enc, mid, tol=tol)
        e_rec = {"layer": mid, "sum_max_abs_diff": ph["sum_max_abs_diff"],
                 "n_heads": ph["n_heads"], "head_dim": ph["head_dim"],
                 "pass": True}
    except IdentityViolation as e:
        e_rec = {"layer": mid, "pass": False, "error": str(e)}

    # F. frozen-RMS DLA additivity, tolerance derived from the fp32 precision
    #    model (never from the measurement). Explicit recorded SKIP when the
    #    embedding anchor is missing — the DLA panel refuses in that case.
    try:
        f_rec = dla_check(r, model, token=-1)
    except DLAUnavailable as e:
        f_rec = {"skipped_no_embed": True, "pass": None, "reason": str(e)}

    # G. attention-pattern path: identity under the eager forward, per-layer
    #    map rows sum to 1, and the post-restore identity probe (V2f).
    try:
        pat = capture_attention_pattern(model, enc, mid, tol=tol)
        g_rec = {"layer": mid,
                 "eager_identity_max_diff": pat["eager_identity_max_diff"],
                 "row_sum_max_dev": pat["row_sum_max_dev"],
                 "post_restore_identity_max_diff": pat["post_restore_identity_max_diff"],
                 "restored_impl": pat["restored_impl"],
                 "pass": (pat["eager_identity_max_diff"] <= tol
                          and pat["row_sum_max_dev"] <= tol
                          and pat["post_restore_identity_max_diff"] <= tol)}
    except (IdentityViolation, RuntimeError) as e:
        g_rec = {"layer": mid, "pass": False, "error": str(e)}

    ok2 = (b_max <= tol and d_max <= tol and (c_max is None or c_max <= tol)
           and e_rec["pass"] and (f_rec["pass"] is not False) and g_rec["pass"])
    e_str = f"{e_rec['sum_max_abs_diff']:.2e}" if e_rec["pass"] else "FAIL"
    f_str = ("skip" if f_rec.get("skipped_no_embed")
             else f"{f_rec['max_abs_diff']:.2e}")
    g_str = (f"{g_rec['row_sum_max_dev']:.2e}" if g_rec["pass"]
             else "FAIL")
    print(f"[{name}] tokens={n_tok} chat_template={use_chat_template}  "
          f"A={a_max:.2e} B={b_max:.2e} "
          f"C={'skipped' if c_max is None else f'{c_max:.2e}'} D={d_max:.2e} "
          f"E={e_str} F={f_str} G={g_str}  "
          f"{'PASS' if ok2 else 'SECONDARY FAIL'}")
    return {
        "prompt_name": name, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "chat_template": use_chat_template, "n_tokens": n_tok,
        "n_layers": r.n_layers, "layer_path": r.layer_path,
        "attn_module": r.attn_name, "mlp_module": r.mlp_name,
        "final_norm_path": norm_path, "tol": tol,
        "A_identity_max_abs_diff_per_layer": a_diffs,
        "A_identity_max_abs_diff": a_max,
        "B_chaining_max_abs_diff": b_max,
        "C_embed_stream_max_abs_diff": c_max,
        "D_lens_anchor_max_abs_diff": d_max,
        "E_per_head": e_rec,
        "F_dla": f_rec,
        "G_attn_pattern": g_rec,
        "all_pass": ok2,
    }


def validate_model(model_id: str, device: str, tol: float = 1e-4,
                   single_prompt: tuple[str, bool] | None = None) -> dict:
    """Load model_id in FORCED fp32, run the battery (or one prompt), write the
    manifest, free the fp32 copy, return the manifest dict. Raises
    IdentityCheckFailed if the primary identity fails (manifest NOT written —
    a failed gate leaves no certifying artifact, only the printed report)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    torch.manual_seed(SEED)  # defensive only — no stochastic op here (see docstring)
    print(f"model={model_id}  device={device}  dtype=float32 (forced)  "
          f"tol={tol:g}  seed={SEED}")
    print(f"torch={torch.__version__}  transformers={transformers.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device).eval()

    if single_prompt is not None:
        battery = (("user_prompt", single_prompt[0], single_prompt[1]),)
    else:
        battery = PROMPT_BATTERY
    print(f"battery: {[b[0] for b in battery]}  "
          f"(fixed constants in selfcheck.py, not sampled)\n")

    try:
        records = [run_checks(model, tokenizer, name, prompt, tmpl, device, tol)
                   for name, prompt, tmpl in battery]
        commit_hash = getattr(model.config, "_commit_hash", None)
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    manifest = {
        "tool": "resid_viewer selfcheck",
        "model": model_id,
        "model_commit_hash": commit_hash,
        "device": device, "forced_dtype": "float32",
        "seed": SEED,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "code_sha256": code_sha256(),
        "battery_is_default": single_prompt is None,
        "checks": records,
        "rule9_inheritance": RULE9_INHERITANCE,
        "shortcuts_taken": [
            "capture path copies every hooked tensor to CPU (.detach().cpu()); "
            "acceptable for single prompts, must move to on-GPU reduction before "
            "any batched/scaled use (comment in capture.py)",
            "prompt battery is 4 fixed hardcoded prompts, not a sampled corpus; "
            "the identity under test is structural (forward code-path property), "
            "so this is code-path coverage, not a statistical sample",
            "validation is per (model family x transformers version); this run "
            "certifies only the model above on the versions above — rerun "
            "selfcheck on any other model before trusting the decomposition",
            "batch size 1 only; batched forward not exercised",
        ],
        # No timestamp on purpose: two runs must be byte-identical (diffable).
    }
    MANIFEST_DIR.mkdir(exist_ok=True)
    path = manifest_path_for(model_id)
    path.write_text(json.dumps(manifest, indent=2) + "\n")

    a_max = max(rec["A_identity_max_abs_diff"] for rec in records)
    ok2 = all(rec["all_pass"] for rec in records)
    print(f"\nRESULT: [A] PASS on all {len(records)} prompts "
          f"(max abs diff {a_max:.6e}); secondary checks "
          f"{'all PASS' if ok2 else 'FAILED — see above'}.")
    print(f"manifest: {path}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=None,
                    help="replace the fixed 4-prompt battery with this single prompt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--chat-template", action="store_true",
                    help="(only with --prompt) wrap it in the chat template")
    args = ap.parse_args()

    single = (args.prompt, args.chat_template) if args.prompt is not None else None
    try:
        manifest = validate_model(args.model, args.device, args.tol, single)
    except IdentityCheckFailed:
        return 1
    return 0 if all(rec["all_pass"] for rec in manifest["checks"]) else 2


if __name__ == "__main__":
    sys.exit(main())
