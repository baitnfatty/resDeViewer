"""Standalone UI server for the residual-stream decomposition viewer.

STANDALONE — imports nothing from ``acsl``; shares no code or state with the
ACSL server (port 5000). Runs on port 5001. Any HF decoder-only LM.

PERMANENT GATE: a model cannot be loaded for analysis unless a valid selfcheck
manifest exists for it under the CURRENT torch/transformers versions and
CURRENT viewer code (see selfcheck.load_valid_manifest). If none exists,
/api/load first runs the full fp32 battery (selfcheck.validate_model) and only
proceeds on PASS. A primary-identity failure refuses the model — no
workaround, per the selfcheck contract.

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
    BASIS_NOTES, ResidualCapture, component_write_norms, cross_layer_cosine,
    find_final_norm, logit_lens)
from resid_viewer.selfcheck import (  # noqa: E402
    DEFAULT_MODEL, IdentityCheckFailed, load_valid_manifest, validate_model)
from resid_viewer.selfcheck import _encode  # noqa: E402

FRONTEND = Path(__file__).resolve().parent / "frontend.html"
DTYPES = {"float32": torch.float32, "float16": torch.float16}

app = FastAPI(title="resid_viewer")


class State:
    lock = threading.Lock()
    model = None
    tokenizer = None
    model_id: str | None = None
    dtype: str | None = None
    gate: dict | None = None       # summary of the manifest that admitted the model
    cap = None                     # last CaptureResult
    tokens: list[str] = []
    final_norm = None


S = State()


class LoadReq(BaseModel):
    model: str = DEFAULT_MODEL
    dtype: str = "float32"
    revalidate: bool = False


class CaptureReq(BaseModel):
    prompt: str
    chat_template: bool = False


class CosineReq(BaseModel):
    component: str = "mlp_out"     # 'mlp_out' | 'attn_out'
    token: int = -1


class LensGridReq(BaseModel):
    state: str = "resid_post"      # 'resid_pre' | 'resid_mid' | 'resid_post'


class LensTopkReq(BaseModel):
    state: str = "resid_post"
    layer: int
    token: int
    top_k: int = 10


def _gate_summary(manifest: dict) -> dict:
    return {
        "model": manifest["model"],
        "model_commit_hash": manifest.get("model_commit_hash"),
        "torch_version": manifest["torch_version"],
        "transformers_version": manifest["transformers_version"],
        "n_prompts": len(manifest["checks"]),
        "A_max_abs_diff": max(c["A_identity_max_abs_diff"] for c in manifest["checks"]),
        "tol": manifest["checks"][0]["tol"],
        "all_pass": all(c["all_pass"] for c in manifest["checks"]),
    }


def _state_tensor(state: str, layer: int) -> torch.Tensor:
    if S.cap is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    if state == "resid_pre":
        return S.cap.resid_pre[layer]
    if state == "resid_mid":
        return S.cap.resid_mid(layer)
    if state == "resid_post":
        return S.cap.resid_post[layer]
    raise HTTPException(400, f"unknown state '{state}'")


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
        "has_capture": S.cap is not None,
        "n_tokens": len(S.tokens), "n_layers": S.cap.n_layers if S.cap else None,
    }


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
            # Free any loaded model first: the fp32 check copy needs the VRAM.
            S.model = None
            S.cap = None
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
        S.cap = None
        if device == "cuda":
            torch.cuda.empty_cache()
        S.tokenizer = AutoTokenizer.from_pretrained(req.model)
        model = AutoModelForCausalLM.from_pretrained(
            req.model, torch_dtype=DTYPES[req.dtype])
        S.model = model.to(device).eval()
        S.final_norm, _ = find_final_norm(S.model)
        S.model_id, S.dtype = req.model, req.dtype
        S.gate = _gate_summary(manifest)
        S.tokens = []
    return {"ok": True, "gate": S.gate, "dtype": req.dtype}


@app.post("/api/capture")
def capture(req: CaptureReq):
    if S.model is None:
        raise HTTPException(400, "no model loaded — POST /api/load first")
    device = next(S.model.parameters()).device
    with S.lock:
        enc = _encode(S.tokenizer, req.prompt, req.chat_template, device)
        with torch.no_grad(), ResidualCapture(S.model) as cap:
            S.model(**enc)
        S.cap = cap.result
        S.tokens = S.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        norms = component_write_norms(S.cap)  # fp32 accumulation inside
    return {
        "tokens": S.tokens, "n_layers": S.cap.n_layers,
        "chat_template": req.chat_template,
        "norms": {k: v.tolist() for k, v in norms.items()},
    }


@app.post("/api/cosine")
def cosine(req: CosineReq):
    if S.cap is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    if req.component not in ("mlp_out", "attn_out"):
        raise HTTPException(400, "component must be 'mlp_out' or 'attn_out'")
    with S.lock:
        m = cross_layer_cosine(S.cap, req.component, req.token)
    return {"matrix": m.tolist(), "component": req.component, "token": req.token}


@app.post("/api/lens_grid")
def lens_grid(req: LensGridReq):
    """Top-1 lens decode for every (layer, token) of the chosen state."""
    if S.cap is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    with S.lock:
        grid = []
        for L in range(S.cap.n_layers):
            ids, probs = logit_lens(_state_tensor(req.state, L), S.model,
                                    S.final_norm, top_k=1)
            toks = S.tokenizer.convert_ids_to_tokens(ids[:, 0].tolist())
            grid.append([{"t": t, "p": round(float(p), 4)}
                         for t, p in zip(toks, probs[:, 0].tolist())])
    return {"state": req.state, "grid": grid}


@app.post("/api/lens_topk")
def lens_topk(req: LensTopkReq):
    if S.cap is None:
        raise HTTPException(400, "no capture yet — POST /api/capture first")
    with S.lock:
        state = _state_tensor(req.state, req.layer)[req.token]
        ids, probs = logit_lens(state, S.model, S.final_norm, top_k=req.top_k)
        toks = S.tokenizer.convert_ids_to_tokens(ids.tolist())
    return {"state": req.state, "layer": req.layer, "token": req.token,
            "topk": [{"t": t, "p": round(float(p), 5)}
                     for t, p in zip(toks, probs.tolist())]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
