#!/usr/bin/env bash
# Start (or stop) the standalone resid_viewer UI server in the pinned ROCm
# container. Port 5001 — deliberately distinct from the ACSL server (5000);
# no shared state, no shared container.
#
#   bash resid_viewer/run_server.sh          # start detached, prints URL
#   bash resid_viewer/run_server.sh --stop   # stop & remove the container
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--stop" ]]; then
  docker rm -f resid-viewer 2>/dev/null && echo "stopped." || echo "not running."
  exit 0
fi

docker rm -f resid-viewer 2>/dev/null || true
docker run -d --name resid-viewer \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$HOME/Projects:/workspace" \
  -v hf_cache:/root/.cache/huggingface \
  -w /workspace/ACSL \
  -e HF_HUB_OFFLINE=1 \
  -p 5001:5001 \
  rig:2.9.1 \
  bash -c "bash scripts/setup_container.sh >/dev/null && python resid_viewer/server.py"
echo "resid_viewer starting — http://localhost:5001 (logs: docker logs -f resid-viewer)"
