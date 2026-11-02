function hpc_worker
root=getenv('ABMF_CODE_DIR');src=fullfile(root,'src');addpath(src);
u=jsondecode(fileread(getenv('ABMF_UNIT_JSON')));output=getenv('ABMF_OUTPUT_DIR');
assert(~exist(output,'dir'),'Fresh output directory required.');mkdir(output);
fprintf('ABMF_STAGE: MATLAB %s; %s; trial %d; nu=%.6g epsilon0=%.6g\n',version,computer,u.trial,u.nu,u.epsilon0);
self_check;
fprintf('ABMF_STAGE: self-check passed; starting original-protocol trial.\n');
if strcmp(u.kind,'maxcut')
    if strcmp(u.profile,'standard')
        conditions={fullfile(src,'data','G55.txt'),50;fullfile(src,'data','G67.txt'),50;fullfile(src,'data','G67.txt'),100};
        trials=10;rounds=100;max_iter=3000;max_full=1200;
    else
        conditions={fullfile(src,'data','tiny_gset.txt'),2};trials=1;rounds=20;max_iter=100;max_full=100;
    end
    run_maxcut_benchmark('condition_list',conditions,'num_trials',trials,'trial_ids',u.trial, ...
        'rounds',rounds,'max_iter',max_iter,'tol',1e-6,'epsilon',1e-6,'gamma0','auto', ...
        'gamma_base',0.25,'rho0',1,'rho_growth',2,'rho_max',1e6,'multiplier_update_period',1, ...
        'constraint_tol',1e-6,'full_sdp',true,'max_full_sdp_n',max_full, ...
        'require_nonnegative_weights',false,'normalize_laplacian',true,'save_histories',true, ...
        'verbose',true,'progress_period',100,'gamma_nu',u.nu,'gamma_mix_epsilon0',u.epsilon0,'output_dir',output);
else
    mode='full';if strcmp(u.profile,'smoke'),mode='quick';end
    overrides=struct('output_dir',output,'trial_ids',u.trial,'gamma_nu',u.nu,'gamma_mix_epsilon0',u.epsilon0);
    if strcmp(u.suite,'sensitivity')
        run_nn_gamma_multiplier_sensitivity(mode,src,overrides);
    else
        run_nonconvex_nn_benchmark(mode,src,overrides);
    end
end
fid=fopen(fullfile(output,'UNIT_DONE.json'),'w');assert(fid>=0);
guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
value=struct('completed',true,'trial',u.trial,'profile',u.profile,'suite',u.suite,'matlab_version',version);
fwrite(fid,unicode2native(jsonencode(value),'UTF-8'),'uint8');
fprintf('ABMF_STAGE: trial returned successfully; controller will audit outputs.\n');
end
