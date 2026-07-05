#!/usr/bin/env bash
# One-command regeneration of the resid_viewer self-check from a host shell.
# Exactly the environment the reported numbers came from: the pinned ROCm
# container (rig:2.9.1), models from the hf_cache volume, offline.
# Extra args are passed through to selfcheck.py (e.g. --model, --tol).
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$HOME/Projects:/workspace" \
  -v hf_cache:/root/.cache/huggingface \
  -w /workspace/ACSL \
  -e HF_HUB_OFFLINE=1 \
  rig:2.9.1 \
  bash -c "bash scripts/setup_container.sh >/dev/null && python resid_viewer/selfcheck.py $*"
