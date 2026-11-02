#!/bin/bash
set -euo pipefail
if ! type module >/dev/null 2>&1; then
    for ABMF_MODULE_INIT in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile; do
        if [[ -f "$ABMF_MODULE_INIT" ]]; then
            set +u; source "$ABMF_MODULE_INIT"; set -u
            if type module >/dev/null 2>&1; then break; fi
        fi
    done
fi
if ! type module >/dev/null 2>&1; then echo 'Environment Modules is unavailable.' >&2;exit 1;fi
set +u
module load matlab-R2022b
set -u
command -v matlab
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec matlab -nodisplay -nosplash -nodesktop -r "try, addpath(getenv('ABMF_CODE_DIR')); hpc_worker; catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0)"
