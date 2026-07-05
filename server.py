"""Standalone UI server for the residual-stream decomposition viewer.

STANDALONE — imports nothing from ``acsl``; shares no code or state with the
ACSL server (port 5000). Runs on port 5001. Any HF decoder-only LM.

PERMANENT GATE: a model cannot be loaded for analysis unless a valid selfcheck
manifest exists for it under the CURRENT torch/transformers versions and
CURRENT viewer code (see selfcheck.load_valid_manifest). If none exists,
/api/load first runs the full fp32 battery (selfcheck.validate_model) and only
proceeds on PASS. A primary-identity failure refuses the model — no
workaround, per the selfcheck contract.

Model-state mutation policy: anything that toggles the attention
implementation lives in capture.py (inside the gate's hash perimeter) and
re-asserts the additive identity after restoring — this file is plumbing only.

PERFORMANCE (prototype): captures copy every tensor to CPU, and the logit-lens
grid pushes each layer's full-vocab logits through the head sequentially —
fine for interactive single prompts, slow for very long ones. The scaled
version should reduce on-GPU and stream results (see capture.py note).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resid_viewer.capture import (  # noqa: E402
    BASIS_NOTES, CaptureResult, DLAUnavailable, IdentityViolation, Intervention,
    PerHeadRefused, ResidualCapture, VRAMRefusal, capture_attention_pattern,
    capture_per_head, component_write_norms, cross_layer_cosine,
    direction_projection, dla_attribution, ensure_vram_headroom, find_final_norm,
    logit_lens, write_geometry, build_session_export, write_session_export)
from resid_viewer.selfcheck import (  # noqa: E402
    DEFAULT_MODEL, MANIFEST_DIR, IdentityCheckFailed, load_valid_manifest,
    manifest_path_for, validate_model)
from resid_viewer.selfcheck import _encode, code_sha256, _file_sha256  # noqa: E402

FRONTEND = Path(__file__).resolve().parent / "frontend.html"
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
DTYPES = {"float32": torch.float32, "float16": torch.float16}

app = FastAPI(title="resid_viewer")


class State:
    lock = threading.Lock()
    model = None
    tokenizer = None
    model_id: str | None = None
    dtype: str | None = None
    gate: dict | None = None       # summary of the manifest that admitted the model
    cap: CaptureResult | None = None
    baseline: CaptureResult | None = None   # un-intervened capture, same prompt
    tokens: list[str] = []
    prompt: str | None = None
    chat_template: bool = False
    enc = None                     # CPU copy of tokenized inputs (re-forwards)
    interventions: list[Intervention] = []
    final_norm = None
    capture_seq: int = 0
    pattern_cache: dict = {}       # (capture_seq, layer) -> pattern dict


S = State()


class LoadReq(BaseModel):
    model: str = DEFAULT_MODEL
    dtype: str = "float32"
    revalidate: bool = False


class InterventionReq(BaseModel):
    layer: int
    component: str   # 'attn' | 'mlp'
    scale: float = 0.0


class CaptureReq(BaseModel):
    prompt: str
    chat_template: bool = False
    interventions: list[InterventionReq] = []
    on_gpu: bool = False   # keep captures on device; guarded, refuses on shortfall


class CosineReq(BaseModel):
    component: str = "mlp_out"
    token: int = -1
    which: str = "current"         # 'current' | 'baseline'


class LensGridReq(BaseModel):
    state: str = "resid_post"
    which: str = "current"


class LensTopkReq(BaseModel):
    state: str = "resid_post"
    layer: int
    token: int
    top_k: int = 10
    which: str = "current"


class HeadsReq(BaseModel):
    layer: int
    token: int = -1
    top_k: int = 3


class DLAReq(BaseModel):
    token: int = -1
    target: str                    # target token TEXT (must be a single token)


class GeometryReq(BaseModel):
    which: str = "current"


class ProjectReq(BaseModel):
    vector: list[float]
    state: str = "resid_post"
    which: str = "current"


class PatternReq(BaseModel):
    layer: int
    head: int


class ExportReq(BaseModel):
    include_raw: bool = False


def _gate_summary(manifest: dict) -> dict:
    checks = manifest["checks"]
    f_recs = [c.get("F_dla", {}) for c in checks]
    return {
        "model": manifest["model"],
        "model_commit_hash": manifest.get("model_commit_hash"),
        "torch_version": manifest["torch_version"],
        "transformers_version": manifest["transformers_version"],
        "n_prompts": len(checks),
        "A_max_abs_diff": max(c["A_identity_max_abs_diff"] for c in checks),
        "tol": checks[0]["tol"],
        "all_pass": all(c["all_pass"] for c in checks),
        "E_pass": all(c.get("E_per_head", {}).get("pass") for c in checks),
        "F_dla_certified": all(f.get("pass") is True for f in f_recs),
        "F_skipped_no_embed": any(f.get("skipped_no_embed") for f in f_recs),
        "G_pass": all(c.get("G_attn_pattern", {}).get("pass") for c in checks),
    }


def _need_model():
    if S.model is None:
        raise HTTPException(400, "no model loaded — POST /api/load first")


def _which_cap(which: str) -> CaptureResult:
    if S.cap is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    if which == "current":
        return S.cap
    if which == "baseline":
        if S.baseline is None:
            raise HTTPException(400, "no baseline — capture with interventions to get one")
        return S.baseline
    raise HTTPException(400, "which must be 'current' or 'baseline'")


def _state_tensor(cap: CaptureResult, state: str, layer: int) -> torch.Tensor:
    if state == "resid_pre":
        return cap.resid_pre[layer]
    if state == "resid_mid":
        return cap.resid_mid(layer)
    if state == "resid_post":
        return cap.resid_post[layer]
    raise HTTPException(400, f"unknown state '{state}'")


def _enc_on_device():
    device = next(S.model.parameters()).device
    return {k: v.to(device) for k, v in S.enc.items()}


def _norms(cap: CaptureResult) -> dict:
    return {k: v.tolist() for k, v in component_write_norms(cap).items()}


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.get("/api/status")
def status():
    import transformers
    return {
        "model_loaded": S.model is not None,
        "model": S.model_id, "dtype": S.dtype,
        "gate": S.gate,
        "basis_notes": BASIS_NOTES,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "attn_implementation": (getattr(S.model.config, "_attn_implementation", None)
                                if S.model is not None else None),
        "has_capture": S.cap is not None,
        "has_baseline": S.baseline is not None,
        "counterfactual": bool(S.interventions),
        "interventions": [vars(i) for i in S.interventions],
        "n_tokens": len(S.tokens), "n_layers": S.cap.n_layers if S.cap else None,
    }


@app.get("/api/manifests")
def manifests():
    """Certification inventory: every manifest on disk with LIVE validity
    against current torch/transformers versions and current code hashes."""
    import json as _json
    out = []
    if MANIFEST_DIR.exists():
        for p in sorted(MANIFEST_DIR.glob("*.json")):
            try:
                m = _json.loads(p.read_text())
            except (ValueError, OSError) as e:
                out.append({"file": p.name, "status": "unreadable", "reason": str(e)})
                continue
            model = m.get("model")
            # Slug guard: a stray/renamed file whose name doesn't re-derive from
            # its own model field would otherwise be listed from one file but
            # validated against another.
            if model is None or manifest_path_for(model).name != p.name:
                out.append({"file": p.name, "model": model, "status": "flagged",
                            "reason": "filename slug does not match manifest's model field"})
                continue
            valid, reason = load_valid_manifest(model)
            out.append({
                "file": p.name, "model": model,
                "model_commit_hash": m.get("model_commit_hash"),
                "status": "valid" if valid else "stale", "reason": reason,
                "A_max_abs_diff": max(
                    (c.get("A_identity_max_abs_diff", float("nan"))
                     for c in m.get("checks", [])), default=None),
                "torch_version": m.get("torch_version"),
                "transformers_version": m.get("transformers_version"),
            })
    return {"manifests": out}


@app.post("/api/load")
def load(req: LoadReq):
    if req.dtype not in DTYPES:
        raise HTTPException(400, f"dtype must be one of {list(DTYPES)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with S.lock:
        # --- permanent gate ---------------------------------------------
        manifest, reason = load_valid_manifest(req.model)
        if manifest is None or req.revalidate:
            print(f"[gate] running fp32 selfcheck for {req.model} "
                  f"(reason: {'revalidate requested' if req.revalidate else reason})")
            S.model = None
            S.cap = S.baseline = None
            S.pattern_cache.clear()
            if device == "cuda":
                torch.cuda.empty_cache()
            try:
                manifest = validate_model(req.model, device)
            except IdentityCheckFailed as e:
                raise HTTPException(422, f"selfcheck identity FAILED — model "
                                         f"refused, no workaround: {e}")
            if not all(c["all_pass"] for c in manifest["checks"]):
                raise HTTPException(422, "selfcheck secondary checks failed — "
                                         "model refused; see server log")
        # --- load for analysis -------------------------------------------
        from transformers import AutoModelForCausalLM, AutoTokenizer
        S.model = None
        S.cap = S.baseline = None
        S.pattern_cache.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
        S.tokenizer = AutoTokenizer.from_pretrained(req.model)
        model = AutoModelForCausalLM.from_pretrained(
            req.model, torch_dtype=DTYPES[req.dtype])
        S.model = model.to(device).eval()
        S.final_norm, _ = find_final_norm(S.model)
        S.model_id, S.dtype = req.model, req.dtype
        S.gate = _gate_summary(manifest)
        S.tokens, S.interventions, S.enc, S.prompt = [], [], None, None
    return {"ok": True, "gate": S.gate, "dtype": req.dtype}


@app.post("/api/capture")
def capture(req: CaptureReq):
    _need_model()
    device = next(S.model.parameters()).device
    ivs = [Intervention(i.layer, i.component, i.scale) for i in req.interventions]
    with S.lock:
        enc = _encode(S.tokenizer, req.prompt, req.chat_template, device)
        on_gpu = req.on_gpu and device.type == "cuda"
        if on_gpu:
            n_cap = 2 if ivs else 1     # baseline doubles the resident cost
            dt = DTYPES[S.dtype]
            est = n_cap * ResidualCapture.estimate_bytes(
                S.model.config.num_hidden_layers,
                enc["input_ids"].shape[1], S.model.config.hidden_size,
                torch.tensor([], dtype=dt).element_size())
            try:
                ensure_vram_headroom(est)
            except VRAMRefusal as e:
                raise HTTPException(507, str(e))
        try:
            baseline = None
            if ivs:
                # Baseline first (same prompt, no hooks), then the intervened
                # run. Baseline is a FULL CaptureResult so every analysis —
                # including ones added later — runs identically on both sides.
                with torch.no_grad(), ResidualCapture(
                        S.model, keep_on_device=on_gpu) as bcap:
                    S.model(**enc)
                baseline = bcap.result
            with torch.no_grad(), ResidualCapture(
                    S.model, interventions=ivs, keep_on_device=on_gpu) as cap:
                S.model(**enc)
        except (IdentityViolation, ValueError) as e:
            raise HTTPException(422, str(e))
        S.cap, S.baseline = cap.result, baseline
        S.interventions = ivs
        S.prompt, S.chat_template = req.prompt, req.chat_template
        S.enc = {k: v.cpu() for k, v in enc.items()}
        S.tokens = S.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        S.capture_seq += 1
        S.pattern_cache.clear()
    return {
        "tokens": S.tokens, "n_layers": S.cap.n_layers,
        "chat_template": req.chat_template,
        "counterfactual": S.cap.is_counterfactual,
        "interventions": [vars(i) for i in ivs],
        "identity_max_diff": S.cap.identity_max_diff() if ivs else None,
        "norms": _norms(S.cap),
        "baseline_norms": _norms(baseline) if baseline is not None else None,
    }


@app.post("/api/cosine")
def cosine(req: CosineReq):
    cap = _which_cap(req.which)
    if req.component not in ("mlp_out", "attn_out"):
        raise HTTPException(400, "component must be 'mlp_out' or 'attn_out'")
    with S.lock:
        m = cross_layer_cosine(cap, req.component, req.token)
    return {"matrix": m.tolist(), "component": req.component, "token": req.token,
            "which": req.which, "counterfactual": cap.is_counterfactual}


@app.post("/api/lens_grid")
def lens_grid(req: LensGridReq):
    """Top-1 lens decode for every (layer, token) of the chosen state."""
    cap = _which_cap(req.which)
    with S.lock:
        grid = []
        for L in range(cap.n_layers):
            ids, probs = logit_lens(_state_tensor(cap, req.state, L), S.model,
                                    S.final_norm, top_k=1)
            toks = S.tokenizer.convert_ids_to_tokens(ids[:, 0].tolist())
            grid.append([{"t": t, "p": round(float(p), 4)}
                         for t, p in zip(toks, probs[:, 0].tolist())])
    return {"state": req.state, "grid": grid, "which": req.which,
            "counterfactual": cap.is_counterfactual}


@app.post("/api/lens_topk")
def lens_topk(req: LensTopkReq):
    cap = _which_cap(req.which)
    with S.lock:
        state = _state_tensor(cap, req.state, req.layer)[req.token]
        ids, probs = logit_lens(state, S.model, S.final_norm, top_k=req.top_k)
        toks = S.tokenizer.convert_ids_to_tokens(ids.tolist())
    return {"state": req.state, "layer": req.layer, "token": req.token,
            "which": req.which, "counterfactual": cap.is_counterfactual,
            "topk": [{"t": t, "p": round(float(p), 5)}
                     for t, p in zip(toks, probs.tolist())]}


@app.post("/api/heads")
def heads(req: HeadsReq):
    """U1: per-head attribution for one layer (on-demand re-forward that
    INHERITS the active intervention spec; refuses on same-layer attn
    intervention — V6c)."""
    _need_model()
    if S.enc is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    with S.lock:
        try:
            ph = capture_per_head(S.model, _enc_on_device(), req.layer,
                                  interventions=S.interventions)
        except PerHeadRefused as e:
            raise HTTPException(409, str(e))
        except (IdentityViolation, IndexError) as e:
            raise HTTPException(422, str(e))
        norms = ph["per_head"].norm(dim=-1)             # fp32 already; [H, T]
        lens = []
        for h in range(ph["n_heads"]):
            ids, probs = logit_lens(ph["per_head"][h][req.token], S.model,
                                    S.final_norm, top_k=req.top_k)
            lens.append([{"t": t, "p": round(float(p), 5)} for t, p in zip(
                S.tokenizer.convert_ids_to_tokens(ids.tolist()), probs.tolist())])
    return {"layer": req.layer, "n_heads": ph["n_heads"], "head_dim": ph["head_dim"],
            "sum_max_abs_diff": ph["sum_max_abs_diff"],
            "has_bias_component": ph["bias"] is not None,
            "norms": norms.tolist(), "token": req.token, "head_lens": lens,
            "counterfactual": ph["counterfactual"],
            "note": "per-head lens decodes each head's raw write via the real "
                    "final norm + lm_head (same convention as the main lens)"}


@app.post("/api/dla")
def dla(req: DLAReq):
    """U2: frozen-RMS direct logit attribution to one target token."""
    cap = _which_cap("current")
    if not (S.gate or {}).get("F_dla_certified"):
        raise HTTPException(409, "DLA is not certified for this model "
                                 "(check F not passed in its manifest) — refusing")
    ids = S.tokenizer.encode(req.target, add_special_tokens=False)
    if len(ids) != 1:
        raise HTTPException(400, f"target {req.target!r} tokenizes to {len(ids)} "
                                 f"tokens; DLA needs a single target token")
    with S.lock:
        try:
            out = dla_attribution(cap, S.model, req.token, ids[0])
            base = (dla_attribution(S.baseline, S.model, req.token, ids[0])
                    if S.baseline is not None else None)
        except DLAUnavailable as e:
            raise HTTPException(409, str(e))
    return {"current": out, "baseline": base, "target": req.target}


@app.post("/api/geometry")
def geometry(req: GeometryReq):
    cap = _which_cap(req.which)
    with S.lock:
        g = {k: v.tolist() for k, v in write_geometry(cap).items()}
    return {"geometry": g, "which": req.which,
            "counterfactual": cap.is_counterfactual}


@app.post("/api/project")
def project(req: ProjectReq):
    cap = _which_cap(req.which)
    with S.lock:
        try:
            p = direction_projection(cap, torch.tensor(req.vector), req.state)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"projection": {k: v.tolist() for k, v in p.items()},
            "state": req.state, "which": req.which,
            "counterfactual": cap.is_counterfactual,
            "note": "dot(state, unit vector) — raw-residual basis"}


@app.post("/api/attn_pattern")
def attn_pattern(req: PatternReq):
    """U3: one layer's QK map under a dedicated eager forward; sdpa restored +
    identity re-asserted afterwards (all inside capture.py — hash perimeter)."""
    _need_model()
    if S.enc is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    with S.lock:
        key = (S.capture_seq, req.layer)
        pat = S.pattern_cache.get(key)
        if pat is None:
            try:
                pat = capture_attention_pattern(S.model, _enc_on_device(),
                                                req.layer,
                                                interventions=S.interventions)
            except (IdentityViolation, IndexError, RuntimeError) as e:
                raise HTTPException(422, str(e))
            S.pattern_cache = {key: pat}     # cache exactly one layer
        w = pat["weights"]
        if not (0 <= req.head < w.shape[0]):
            raise HTTPException(400, f"head {req.head} out of range [0,{w.shape[0]})")
    return {"layer": req.layer, "head": req.head, "n_heads": int(w.shape[0]),
            "map": w[req.head].tolist(),
            "row_sum_max_dev": pat["row_sum_max_dev"],
            "eager_identity_max_diff": pat["eager_identity_max_diff"],
            "post_restore_identity_max_diff": pat["post_restore_identity_max_diff"],
            "restored_impl": pat["restored_impl"],
            "counterfactual": pat["counterfactual"],
            "note": "map from a separate eager forward; other panels use the "
                    "default forward (see basis notes)"}


@app.post("/api/export")
def export(req: ExportReq):
    """U7: session export — every number on screen becomes a file on disk."""
    cap = _which_cap("current")
    import hashlib as _hl
    with S.lock:
        here = Path(__file__).resolve().parent
        meta = {
            "model": S.model_id, "model_commit_hash": (S.gate or {}).get("model_commit_hash"),
            "dtype": S.dtype,
            "device": str(next(S.model.parameters()).device),
            "prompt": S.prompt,
            "prompt_sha256": _hl.sha256((S.prompt or "").encode()).hexdigest(),
            "chat_template": S.chat_template,
            "tokens": S.tokens,
            "interventions": [vars(i) for i in S.interventions],
            "counterfactual": cap.is_counterfactual,
            "gate_manifest": manifest_path_for(S.model_id).name,
            # code_state records MORE than the gate perimeter (which stays
            # capture.py+selfcheck.py): server.py also shaped these bytes.
            "code_state": {**code_sha256(),
                           "server.py": _file_sha256(here / "server.py")},
            "reproducibility": "deterministic per (machine, dtype); fp16 "
                               "analyses are byte-stable only on the same "
                               "hardware/kernels",
        }
        analyses = {"norms": _norms(cap),
                    "geometry": {k: v.tolist() for k, v in write_geometry(cap).items()},
                    "cosine_mlp_last_tok": cross_layer_cosine(cap, "mlp_out", -1).tolist(),
                    "cosine_attn_last_tok": cross_layer_cosine(cap, "attn_out", -1).tolist()}
        if S.baseline is not None:
            analyses["baseline_norms"] = _norms(S.baseline)
        payload = build_session_export(meta, analyses)
        path = write_session_export(CAPTURES_DIR, payload)
        raw_path = None
        if req.include_raw:
            import numpy as np
            raw_path = path.with_suffix(".npz")
            np.savez_compressed(
                raw_path,
                **{f"{n}_{L}": getattr(cap, n)[L].float().numpy()
                   for n in ("resid_pre", "attn_out", "mlp_out", "resid_post")
                   for L in range(cap.n_layers)})
    return {"path": str(path), "raw_path": str(raw_path) if raw_path else None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
