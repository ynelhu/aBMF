#!/bin/bash
# Submit the full main experiment directly; no separate smoke submission.
set -euo pipefail
ABMF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for ABMF_ARGUMENT in "$@"; do
    case "$ABMF_ARGUMENT" in
        --profile|--profile=*|--suite|--suite=*)
            echo 'RUN_FORMAL.sh always runs the full main experiment; profile/suite selectors are not accepted.' >&2
            exit 2
            ;;
    esac
done
exec bash "$ABMF_ROOT/START_ALL.sh" --profile standard --suite main "$@"

