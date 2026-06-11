#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate metamMuon
export PYTHONPATH="${REPO_ROOT}"

echo "=== Nightly bound search: propose next trial ==="
python scripts/nightly_bound_search.py propose

echo
echo "=== Search status ==="
python scripts/nightly_bound_search.py status
