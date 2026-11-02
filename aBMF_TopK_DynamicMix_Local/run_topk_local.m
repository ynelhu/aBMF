function [summary,result_zip] = run_topk_local(profile,penalty)
%RUN_TOPK_LOCAL Windows MATLAB one-click run with self-check, statistics and ZIP.
if nargin<1,profile='standard';end
if nargin<2,penalty=struct();end
if ~isfield(penalty,'nu'),penalty.nu=0.9;end
if ~isfield(penalty,'epsilon0'),penalty.epsilon0=1e-6;end
if ~isfield(penalty,'gamma0'),penalty.gamma0='auto';end
assert(isscalar(penalty.nu)&&isfinite(penalty.nu)&&penalty.nu>0&&penalty.nu<1,'Require 0<nu<1.');
assert(isscalar(penalty.epsilon0)&&isfinite(penalty.epsilon0)&&penalty.epsilon0>0,'epsilon0 must be positive.');
profile=lower(char(profile));
switch profile
    case 'standard'
        fixed_n=50000;rank_list=1000;fixed_rank=50;n_list=10000;num_trials=5;max_iter=1000;
    case 'local'
        fixed_n=1000;rank_list=20;fixed_rank=5;n_list=500;num_trials=5;max_iter=1000;
    case 'smoke'
        fixed_n=200;rank_list=5;fixed_rank=3;n_list=100;num_trials=1;max_iter=30;
    otherwise
        error('topk:profile','Use standard, local, or smoke.');
end
root=fileparts(mfilename('fullpath'));addpath(root);
parent=fullfile(root,'results');if ~exist(parent,'dir'),mkdir(parent);end
[~,tag]=fileparts(tempname(parent));tag=tag(max(1,numel(tag)-7):end);
run_name=['topk_mix_' profile '_' datestr(now,'yyyymmddTHHMMSS') '_' tag];
output=fullfile(parent,run_name);assert(~exist(output,'dir'),'Use a fresh result directory.');mkdir(output);
code=fullfile(output,'code_snapshot');mkdir(code);
names={'RUN_LOCAL.m','run_topk_local.m','run_topk_benchmark.m','gamma_mix_topk.m', ...
    'CHECK_GAMMA_RULE.m','AUDIT_GAMMA_HISTORY.m','README_CN.txt'};
for j=1:numel(names),copyfile(fullfile(root,names{j}),code);end
settings=struct('nu',penalty.nu,'epsilon0',penalty.epsilon0,'sigma_h',0);
run_info=struct('profile',profile,'protocol','table2_two_conditions_v1', ...
    'methods',{{'sBMF-GD','sBMF-GN','aBMF-adp','aBMF-our'}}, ...
    'gamma_rule','gamma_k=max(min(gamma_{k-1},hat_gamma_k),check_gamma_k)', ...
    'penalty',penalty,'sigma_h',0,'gradient','X_k*Y_k''-A','matrix_norm','Frobenius, exact low-rank identity', ...
    'update_timing','Compute candidates after both blocks at X_k,Y_k; use updated value for next pair.', ...
    'fixed_n',fixed_n,'rank_list',rank_list,'fixed_rank',fixed_rank,'n_list',n_list, ...
    'num_trials',num_trials,'max_iter',max_iter,'tol',1e-6,'seed',1,'matrix_types',{{'PSD'}}, ...
    'data_mode','matrix_free','matlab_version',version,'computer',computer, ...
    'expected_trial_conditions',(numel(rank_list)+numel(n_list))*num_trials, ...
    'expected_method_records',(numel(rank_list)+numel(n_list))*num_trials*4,'started_local_time',datestr(now,31));
write_local_json(fullfile(output,'LOCAL_RUN.json'),run_info);
diary(fullfile(output,'execution.log'));diary_guard=onCleanup(@()diary('off')); %#ok<NASGU>
clock_start=tic;phase='self_check';
try
    fprintf('Running small dense-reference and solver-interface checks first.\n');
    self_check=CHECK_GAMMA_RULE;
    write_local_json(fullfile(output,'SELF_CHECK.json'),self_check);
    phase='experiment';
    fprintf('\nProfile: %s; nu=%.6g; epsilon0=%.6g; sigma_h=0.\n',profile,penalty.nu,penalty.epsilon0);
    fprintf('Output: %s\nExpected method records: %d\n',output,run_info.expected_method_records);
    summary=run_topk_benchmark('fixed_n',fixed_n,'rank_list',rank_list,'fixed_rank',fixed_rank,'n_list',n_list, ...
        'matrix_types',{'PSD'},'num_trials',num_trials,'seed',1,'max_iter',max_iter,'tol',1e-6, ...
        'our_gamma0',penalty.gamma0,'nu',penalty.nu,'epsilon0',penalty.epsilon0, ...
        'epsilon',1e-6,'gamma0','auto','gamma_max',1e8,'gn_rank_floor',1e-10, ...
        'data_mode','matrix_free','explicit_threshold',2000,'eigs_tol',1e-10,'eigs_maxit',2000, ...
        'reference_tol',1e-12,'reference_maxit',4000,'save_histories',true, ...
        'include_projection',false,'save_explicit_matrices',false,'make_plots',true,'output_dir',output);
    phase='history_audit';
    [checks,raw]=AUDIT_GAMMA_HISTORY(output,settings,run_info.expected_trial_conditions,run_info.expected_method_records);
    writetable(struct2table(checks),fullfile(output,'gamma_check_summary.csv'));
    our=raw(strcmp(raw.method,'aBMF-our'),:);
    handoff=struct('status','completed','partial',false,'profile',profile,'method_records',height(raw), ...
        'expected_method_records',run_info.expected_method_records,'self_check_passed',true,'gamma_check_passed',true, ...
        'our_histories_checked',numel(checks),'our_iterations_checked',sum([checks.iterations]), ...
        'our_joint_tolerance_count',sum(strcmp(string(our.stop_reason),'joint_tolerance')), ...
        'our_iteration_limit_count',sum(strcmp(string(our.stop_reason),'iteration_limit')), ...
        'penalty',penalty,'sigma_h',0,'gamma_rule',run_info.gamma_rule,'total_wall_seconds',toc(clock_start), ...
        'finished_local_time',datestr(now,31),'note','Mixed update is a practical mechanism; completion is not an exactness or stationarity certificate.');
    write_local_json(fullfile(output,'HANDOFF.json'),handoff);
    fprintf('\nCOMPLETED: %d records; our history audits passed.\n',height(raw));
    diary('off');phase='result_pack';
    result_zip=fullfile(parent,[run_name '_RESULTS.zip']);zip(result_zip,{run_name},parent);
    fprintf('Result ZIP: %s\n',result_zip);
catch ME
    fprintf(2,'%s\n',getReport(ME,'extended','hyperlinks','off'));
    handoff=struct('status','failed_or_interrupted','partial',true,'phase',phase,'profile',profile, ...
        'expected_method_records',run_info.expected_method_records,'gamma_check_passed',false, ...
        'error_identifier',ME.identifier,'error_message',ME.message,'total_wall_seconds',toc(clock_start));
    write_local_json(fullfile(output,'HANDOFF.json'),handoff);diary('off');
    try
        diagnostic_zip=fullfile(parent,[run_name '_PARTIAL.zip']);zip(diagnostic_zip,{run_name},parent);
        fprintf(2,'Diagnostic ZIP: %s\n',diagnostic_zip);
    catch pack_error
        fprintf(2,'Could not zip diagnostics: %s\nFiles remain in %s\n',pack_error.message,output);
    end
    rethrow(ME);
end
end

function write_local_json(filename,value)
fid=fopen(filename,'w');assert(fid>=0,'Cannot write %s.',filename);
guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,unicode2native(jsonencode(value),'UTF-8'),'uint8');
end
