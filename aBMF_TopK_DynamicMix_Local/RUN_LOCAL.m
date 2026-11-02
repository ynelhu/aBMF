% Windows MATLAB R2020a: open this file and click Run (F5).
% 'standard' runs only the two Table 2 groups, each repeated five times.
profile = 'standard';  % Alternatives: 'smoke', 'local'.

penalty.nu = 0.9;            % Must satisfy 0 < nu < 1.
penalty.epsilon0 = 1e-6;     % Positive offset in the directional candidate.
penalty.gamma0 = 'auto';     % Positive scalar or original residual-based initialization.

topk_code_dir=fileparts(mfilename('fullpath'));
addpath(topk_code_dir);
clear run_topk_local run_topk_benchmark gamma_mix_topk CHECK_GAMMA_RULE AUDIT_GAMMA_HISTORY
[summary,result_zip]=run_topk_local(profile,penalty);
fprintf('\nResult pack to upload: %s\n',result_zip);
