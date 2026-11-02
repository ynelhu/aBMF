#!/bin/bash
set -euo pipefail
# Slurm copies this script to its spool directory: use the explicit run argument.
if [[ "$#" -ne 2 ]]; then echo 'Expected absolute Python executable and run directory.' >&2; exit 2; fi
ABMF_RUN="$(cd -- "$2" && pwd)"
ABMF_CODE="$ABMF_RUN/code_snapshot"
if [[ ! -f "$ABMF_CODE/hpc/cu_hpc.py" ]]; then
    echo "Missing run snapshot controller: $ABMF_CODE/hpc/cu_hpc.py" >&2; exit 2
fi
echo "ABMF_STAGE: controller=$ABMF_CODE/hpc/cu_hpc.py"
exec "$1" "$ABMF_CODE/hpc/cu_hpc.py" pack "$ABMF_RUN"
