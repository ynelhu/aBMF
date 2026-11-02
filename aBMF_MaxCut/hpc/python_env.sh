#!/bin/bash
set -euo pipefail
if [[ -n "${PYTHON_BIN:-}" ]]; then ABMF_PYTHON="$PYTHON_BIN"
elif [[ -x "$HOME/miniconda3/bin/python3" ]]; then ABMF_PYTHON="$HOME/miniconda3/bin/python3"
elif [[ -x "$HOME/minconda3/bin/python3" ]]; then ABMF_PYTHON="$HOME/minconda3/bin/python3"
else ABMF_PYTHON="$(command -v python3 || true)"; fi
if [[ -z "$ABMF_PYTHON" ]]; then echo 'Python not found; set PYTHON_BIN.' >&2;exit 1;fi
"$ABMF_PYTHON" -c 'import sys; assert sys.version_info>=(3,8), "Use Miniconda Python >=3.8; system Python 3.6 is unsupported."'
export ABMF_PYTHON
