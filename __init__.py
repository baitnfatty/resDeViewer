"""resid_viewer — standalone residual-stream decomposition viewer.

STANDALONE mechanistic-analysis tool. Not an ACSL component: imports nothing
from ``acsl`` (no axes, directions, probe logic, or served-path code), and
nothing in ACSL imports it. Works on any HuggingFace decoder-only LM.
"""
from resid_viewer.capture import (  # noqa: F401
    BASIS_NOTES,
    CaptureResult,
    ResidualCapture,
    component_write_norms,
    cross_layer_cosine,
    find_decoder_layers,
    find_final_norm,
    logit_lens,
)
