#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
READY_FILE="${REPO_ROOT}/experiments/nightly_bound_search/READY_FOR_MORNING.json"

if [[ ! -f "${READY_FILE}" ]]; then
  echo "No morning training proposal found at ${READY_FILE}" >&2
  echo "Run: python scripts/nightly_bound_search.py propose" >&2
  exit 1
fi

TRAIN_CMD="$(python - <<'PY'
import json
from pathlib import Path
ready = Path("experiments/nightly_bound_search/READY_FOR_MORNING.json")
print(json.loads(ready.read_text())["train_command"])
PY
)"

echo "Starting morning training run:"
echo "${TRAIN_CMD}"
eval "${TRAIN_CMD}"
