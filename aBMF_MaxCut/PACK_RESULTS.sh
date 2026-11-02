#!/bin/bash
set -euo pipefail
ABMF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ABMF_ROOT/hpc/python_env.sh"
exec "$ABMF_PYTHON" "$ABMF_ROOT/hpc/cu_hpc.py" pack "$@"
