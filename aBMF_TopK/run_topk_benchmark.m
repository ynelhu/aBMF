function summary = run_topk_benchmark(varargin) % Top-k with the directional/norm mixed gamma update for aBMF-our.
%RUN_TOPK_BENCHMARK Compare SDP projection, sBMF-GD, sBMF-GN, aBMF-our, and aBMF-adp. % Describe the benchmark purpose.
% The solvers receive only the observed matrix A or a matrix-times-block oracle. % State the leakage-free solver interface.
% The planted eigensystem is never returned by the data generator or passed to any solver. % State the spectral-information isolation rule.
% Example: run_topk_benchmark('fixed_n',1000,'rank_list',[2 5 10 20],'fixed_rank',5,'n_list',[500 1000 2000]). % Give a usage example.
ip = inputParser; % Create an input parser for all benchmark options.
addParameter(ip,'fixed_n',1000,@(x)isnumeric(x)&&isscalar(x)&&x>=2); % Add the fixed matrix dimension used in the rank sweep.
addParameter(ip,'rank_list',[2 20],@(x)isnumeric(x)&&isvector(x)&&all(x>=1)); % Add the target ranks used in the fixed-n sweep.
addParameter(ip,'fixed_rank',5,@(x)isnumeric(x)&&isscalar(x)&&x>=1); % Add the fixed target rank used in the dimension sweep.
addParameter(ip,'n_list',[500 2000],@(x)isnumeric(x)&&isvector(x)&&all(x>=2)); % Add the matrix dimensions used in the fixed-rank sweep.
addParameter(ip,'matrix_types',{'PSD','Indefinite'},@(x)iscell(x)||isstring(x)); % Add the symmetric matrix types included in the benchmark.
addParameter(ip,'num_trials',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1); % Add the number of random trials per experimental condition.
addParameter(ip,'seed',1,@(x)isnumeric(x)&&isscalar(x)); % Add the base random seed.
addParameter(ip,'max_iter',500,@(x)isnumeric(x)&&isscalar(x)&&x>=1); % Add the maximum number of iterations for iterative methods.
addParameter(ip,'tol',1e-6,@(x)isnumeric(x)&&isscalar(x)&&x>0); % Add the relative stopping tolerance.
addParameter(ip,'epsilon',1e-6,@(x)isnumeric(x)&&isscalar(x)&&x>0); % Original adp initialization safeguard.
addParameter(ip,'gamma0','auto'); % Original adp initial penalty.
addParameter(ip,'our_gamma0','auto'); % Positive initial our penalty, or auto.
addParameter(ip,'nu',0.9,@(x)isnumeric(x)&&isscalar(x)&&isfinite(x)&&x>0&&x<1);
addParameter(ip,'epsilon0',1e-6,@(x)isnumeric(x)&&isscalar(x)&&isfinite(x)&&x>0);
addParameter(ip,'gamma_max',1e8,@(x)isnumeric(x)&&isscalar(x)&&x>0); % Add the maximum permitted aBMF-adp penalty.
addParameter(ip,'gn_rank_floor',1e-10,@(x)isnumeric(x)&&isscalar(x)&&x>0); % Add the numerical rank safeguard for Gauss-Newton.
addParameter(ip,'data_mode','auto',@(x)ischar(x)||isstring(x)); % Add the observed-data representation mode.
addParameter(ip,'explicit_threshold',2000,@(x)isnumeric(x)&&isscalar(x)&&x>=2); % Add the largest dimension stored as an explicit dense matrix in automatic mode.
addParameter(ip,'eigs_tol',1e-10,@(x)isnumeric(x)&&isscalar(x)&&x>0); % Add the tolerance used by the SDP projection eigensolver.
addParameter(ip,'eigs_maxit',2000,@(x)isnumeric(x)&&isscalar(x)&&x>=1); % Add the maximum iterations used by the SDP projection eigensolver.
addParameter(ip,'reference_tol',1e-12,@(x)isnumeric(x)&&isscalar(x)&&x>0); % Add the tighter tolerance used by the independent reference eigensolver.
addParameter(ip,'reference_maxit',4000,@(x)isnumeric(x)&&isscalar(x)&&x>=1); % Add the maximum iterations used by the independent reference eigensolver.
addParameter(ip,'save_histories',false,@(x)islogical(x)||ismember(x,[0 1])); % Add the option to save iteration histories.
addParameter(ip,'save_explicit_matrices',false,@(x)islogical(x)||ismember(x,[0 1])); % Add the option to save explicit observed matrices for auditing.
addParameter(ip,'output_dir',fullfile(fileparts(mfilename('fullpath')),'results_topk_benchmark')); % Add the output directory option.
addParameter(ip,'trial_ids',[],@(x)isnumeric(x)&&(isempty(x)||isvector(x)));
addParameter(ip,'active_sweeps',{'fixed_n_rank_sweep','fixed_rank_n_sweep'},@iscell);
addParameter(ip,'make_plots',true,@(x)islogical(x)&&isscalar(x));
addParameter(ip,'include_projection',false,@(x)islogical(x)&&isscalar(x)); % Table 2 compares only four factorized methods.
parse(ip,varargin{:}); % Parse all benchmark options.
opt = ip.Results; % Store the parsed options in a structure.
opt.rank_list = unique(round(opt.rank_list(:)')); % Sort and deduplicate the rank sweep values.
opt.n_list = unique(round(opt.n_list(:)')); % Sort and deduplicate the dimension sweep values.
opt.matrix_types = cellstr(opt.matrix_types); % Convert matrix-type labels to a cell array of character vectors.
opt.data_mode = char(opt.data_mode); % Convert the data-mode label to a character vector.
validate_options(opt); % Validate all benchmark dimensions, ranks, and modes.
if isempty(opt.trial_ids), opt.trial_ids=1:opt.num_trials; end
assert(all(isfinite(opt.trial_ids))&&all(opt.trial_ids==fix(opt.trial_ids))&&all(opt.trial_ids>=1)&&all(opt.trial_ids<=opt.num_trials),'Invalid trial_ids.');
opt.trial_ids=unique(opt.trial_ids(:)');
assert(~isempty(opt.active_sweeps)&&all(ismember(opt.active_sweeps,{'fixed_n_rank_sweep','fixed_rank_n_sweep'})),'Invalid active_sweeps.');
if ~exist(opt.output_dir,'dir') % Check whether the benchmark output directory exists.
    mkdir(opt.output_dir); % Create the benchmark output directory when necessary.
end % Finish the output-directory creation block.
raw_rows = repmat(result_row_template(),0,1); % Initialize an empty result array with a fixed field layout.
row_counter = 0; % Initialize the number of stored result rows.
for type_index = 1:numel(opt.matrix_types) % Loop over PSD and indefinite matrix cases.
    matrix_type = opt.matrix_types{type_index}; % Read the current matrix type.
    if any(strcmp('fixed_n_rank_sweep',opt.active_sweeps))
    for rank_index = 1:numel(opt.rank_list) % Loop over target ranks in the fixed-n sweep.
        current_rank = opt.rank_list(rank_index); % Read the current target rank.
        [raw_rows,row_counter] = run_condition(raw_rows,row_counter,'fixed_n_rank_sweep',opt.fixed_n,current_rank,matrix_type,opt); % Run one fixed-n varying-rank condition.
    end % Finish the fixed-n rank-sweep loop.
    end
    if any(strcmp('fixed_rank_n_sweep',opt.active_sweeps))
    for n_index = 1:numel(opt.n_list) % Loop over matrix dimensions in the fixed-rank sweep.
        current_n = opt.n_list(n_index); % Read the current matrix dimension.
        [raw_rows,row_counter] = run_condition(raw_rows,row_counter,'fixed_rank_n_sweep',current_n,opt.fixed_rank,matrix_type,opt); % Run one fixed-rank varying-n condition.
    end % Finish the fixed-rank dimension-sweep loop.
    end
end % Finish the matrix-type loop.
raw_table = struct2table(raw_rows);
writetable(raw_table,fullfile(opt.output_dir,'all_trial_results.csv')); % Preserve trial-level evidence for independent aggregation.
summary_with_sweep = aggregate_results(raw_table); % Compute one mean and one standard deviation for each condition, method, and metric.
fixed_n_summary = select_comparison_columns(summary_with_sweep(strcmp(summary_with_sweep.sweep,'fixed_n_rank_sweep'),:)); % Select the fixed-n varying-rank aggregate comparison.
writetable(fixed_n_summary,fullfile(opt.output_dir,'fixed_n_vary_rank_summary.csv')); % Save only aggregate fixed-n varying-rank statistics.
fixed_rank_summary = select_comparison_columns(summary_with_sweep(strcmp(summary_with_sweep.sweep,'fixed_rank_n_sweep'),:)); % Select the fixed-rank varying-n aggregate comparison.
writetable(fixed_rank_summary,fullfile(opt.output_dir,'fixed_rank_vary_n_summary.csv')); % Save only aggregate fixed-rank varying-n statistics.
summary = combine_unique_summaries(fixed_n_summary,fixed_rank_summary); % Combine both sweeps while retaining only one row for any overlapping n-r condition.
writetable(summary,fullfile(opt.output_dir,'summary_mean_std.csv')); % Save the aggregate-only comparison table.
if opt.make_plots
    try
        plot_sweep_metrics(fixed_n_summary,'rank',fullfile(opt.output_dir,'figures_fixed_n'));
        plot_sweep_metrics(fixed_rank_summary,'n',fullfile(opt.output_dir,'figures_fixed_rank'));
        write_json(fullfile(opt.output_dir,'PLOT_STATUS.json'),struct('created',true,'formats',{{'png','pdf'}}));
    catch plot_error
        write_json(fullfile(opt.output_dir,'PLOT_STATUS.json'),struct('created',false,'error',plot_error.message));
        warning('topk:plot','Numerical tables saved; some figures could not be generated: %s',plot_error.message);
    end
end
config = opt; % Copy the parsed options into the saved configuration structure.
config.methods = {'sBMF-GD','sBMF-GN','aBMF-adp','aBMF-our'};
if opt.include_projection,config.methods=[{'SDP-Projection'},config.methods];end
config.metrics = {'iterations','time','final_gamma','data_objective','rel_objective','subspace_error','rel_disagreement'}; % Record exactly the metrics retained in the aggregate comparison tables.
config.data_isolation = 'Solvers receive only explicit A or an A-times-block function handle; planted Q and planted eigenvalues are not returned.'; % Record the leakage-free interface.
config.reference_isolation = 'The reference eigenspace is recomputed from the observed data and is used only after each solver terminates.'; % Record the reference-evaluation separation.
config.sdp_projection = 'The SDP projection baseline independently recomputes the leading eigenpairs from the observed data.'; % Record the projection implementation.
if ~opt.include_projection,config.sdp_projection='Not run as a baseline. Independent evaluation-reference eigenpairs are still computed.';end
config.indefinite_reference = 'Best rank-r PSD approximation obtained by retaining the r largest positive eigenvalues recomputed from observed data.'; % Explain the indefinite reference.
config.gauss_newton_reference = 'Liu, Wen, and Zhang, An Efficient Gauss-Newton Algorithm for Symmetric Low-Rank Product Matrix Approximations.'; % Record the Gauss-Newton reference.
config.abmf_our_gamma_update = 'gamma_k=max(min(gamma_{k-1},hat_gamma_k),check_gamma_k); retain previous exactly when D_k=0.';
config.abmf_our_gradient = 'G_k=X_k*Y_k''-A; sigma_h=0 because h=0 in this Top-k loss.';
config.abmf_our_norm = 'Exact Frobenius norm via Gram matrices and cached operator products; no n-by-n gradient.';
config.abmf_our_update_timing = 'Blocks producing X_k,Y_k use gamma_{k-1}; candidates at X_k,Y_k give gamma_k for the next block pair. History gamma is used; gamma_next is updated.';
config.abmf_our_directional_denominator = '2*nu*norm(X_k-Y_k,fro)^2; implemented using normalized D.';
config.sigma_h = 0;
config.abmf_our_final_gamma = 'final_gamma is actually used in the last block pair; final_gamma_next is the mixed update at the last factors.';
config.abmf_adp_final_gamma = 'Original convention retained: value after the last adaptive update.';
config.common_evaluation = 'Every method is evaluated through W=(X+Y)/2 and Z=W*W'', with common data objective, relative objective, subspace error, and rel_disagreement=norm(X-Y,''fro'')/max([1,norm(X,''fro''),norm(Y,''fro'')]).'; % Record the unified cross-method evaluation rule.
write_json(fullfile(opt.output_dir,'config.json'),config); % Save the benchmark configuration with old-MATLAB compatibility.
disp(fixed_n_summary); % Display the fixed-n varying-rank aggregate table.
disp(fixed_rank_summary); % Display the fixed-rank varying-n aggregate table.
fprintf('All benchmark results were saved to: %s\n',opt.output_dir); % Print the final output directory.
end % Finish the main benchmark function.

function validate_options(opt) % Validate all benchmark size, rank, and representation settings.
if any(opt.rank_list>=opt.fixed_n) % Check whether any varying rank is invalid for eigs.
    error('Every value in rank_list must be strictly smaller than fixed_n.'); % Stop when a fixed-n rank is invalid.
end % Finish the fixed-n sweep validation.
if any(opt.fixed_rank>=opt.n_list) % Check whether the fixed rank is invalid for any dimension.
    error('fixed_rank must be strictly smaller than every value in n_list.'); % Stop when a fixed-rank dimension is invalid.
end % Finish the fixed-rank sweep validation.
valid_types = {'PSD','Indefinite'}; % Define the supported symmetric matrix types.
for index = 1:numel(opt.matrix_types) % Loop over all requested matrix types.
    if ~any(strcmpi(opt.matrix_types{index},valid_types)) % Check whether the current matrix type is supported.
        error('matrix_types may contain only ''PSD'' and ''Indefinite''.'); % Stop when an unsupported matrix type is requested.
    end % Finish the matrix-type validation test.
end % Finish the matrix-type validation loop.
valid_modes = {'auto','explicit','matrix_free'}; % Define the supported observed-data representation modes.
if ~any(strcmpi(opt.data_mode,valid_modes)) % Check whether the requested data mode is supported.
    error('data_mode must be ''auto'', ''explicit'', or ''matrix_free''.'); % Stop when an unsupported data mode is requested.
end % Finish the data-mode validation test.
end % Finish the option-validation function.

function [raw_rows,row_counter] = run_condition(raw_rows,row_counter,sweep_name,n,r,matrix_type,opt) % Run all trials and methods for one benchmark condition.
condition_name = sprintf('%s_n%d_r%d_%s',sweep_name,n,r,lower(matrix_type)); % Build a unique condition name for saved files.
condition_directory = fullfile(opt.output_dir,'histories',condition_name); % Build the optional per-condition history directory.
if opt.save_histories && ~exist(condition_directory,'dir') % Check whether history saving requires a new directory.
    mkdir(condition_directory); % Create the per-condition history directory.
end % Finish the history-directory creation block.
for trial = opt.trial_ids % Preserve the original trial number and all original seed formulas when splitting HPC jobs.
    problem_seed = opt.seed+100000*strcmpi(matrix_type,'Indefinite')+1000*n+10*r+trial; % Construct a deterministic problem seed.
    initialization_seed = problem_seed+500000; % Construct a separate deterministic initialization seed.
    reference_seed = problem_seed+700000; % Construct a deterministic reference-eigensolver seed.
    projection_seed = problem_seed+900000; % Construct a deterministic SDP-projection eigensolver seed.
    rng(problem_seed,'twister'); % Set the seed used to generate the observed symmetric matrix.
    observed = make_observed_data(n,r,matrix_type,opt.data_mode,opt.explicit_threshold); % Generate only the observed data representation without returning the planted eigensystem.
    solver_data = make_solver_data(observed); % Build the restricted interface passed to every solver.
    verify_solver_interface(solver_data); % Verify that no planted spectral fields are exposed to solvers.
    if opt.save_explicit_matrices && strcmpi(solver_data.mode,'explicit') % Check whether the explicit observed matrix should be archived.
        matrix_directory = fullfile(opt.output_dir,'observed_matrices'); % Build the observed-matrix archive directory.
        if ~exist(matrix_directory,'dir') % Check whether the matrix archive directory exists.
            mkdir(matrix_directory); % Create the matrix archive directory when necessary.
        end % Finish the matrix-directory creation block.
        A = solver_data.A; %#ok<NASGU> % Copy the observed matrix to a plainly named variable for the MAT file.
        save(fullfile(matrix_directory,sprintf('%s_trial_%02d.mat',condition_name,trial)),'A','problem_seed','matrix_type','n','r','-v7.3'); % Save only the observed matrix and public metadata.
        clear A; % Remove the temporary explicit matrix alias after saving.
    end % Finish the optional explicit-matrix archive block.
    rng(initialization_seed,'twister'); % Set the common initialization seed for all factorized methods.
    X0 = randn(n,r)/sqrt(n); % Generate the common initial low-rank factor.
    auto_gamma = max(0.5*residual_fro(solver_data,X0,X0),opt.epsilon); % Compute the automatic initial penalty from observed data and the common initialization.
    gamma0 = max(resolve_gamma(opt.gamma0,auto_gamma),opt.epsilon); % Resolve the initial penalty and enforce the positive lower bound.
    if opt.include_projection
        [solution1,history1] = sdp_projection_solver(solver_data,r,opt.eigs_tol,opt.eigs_maxit,projection_seed);
    end
    [solution2,history2] = symmetric_gd_solver(solver_data,X0,opt.max_iter,opt.tol); % Run symmetric BMF with gradient descent using observed data only.
    [solution3,history3] = symmetric_gauss_newton_solver(solver_data,X0,opt.max_iter,opt.tol,opt.gn_rank_floor); % Run symmetric BMF with Gauss-Newton using observed data only.
    our_gamma0=resolve_gamma(opt.our_gamma0,max(auto_gamma,opt.epsilon0));
    gamma_settings=struct('nu',opt.nu,'epsilon0',opt.epsilon0,'sigma_h',0);
    [solution4,history4] = mixed_alternating_solver(solver_data,X0,opt.max_iter,opt.tol,our_gamma0,gamma_settings);
    [solution5,history5] = alternating_solver('aBMF-adp','adp',solver_data,X0,opt.max_iter,opt.tol,gamma0,opt.epsilon,opt.gamma_max); % Run aBMF with the comparison adaptive update.
    solutions = {solution2,solution3,solution5,solution4};
    if opt.include_projection,solutions=[{solution1},solutions];end
    reference = compute_reference(solver_data,r,opt.reference_tol,opt.reference_maxit,reference_seed); % Recompute the reference only after every optimization method has terminated.
    for method_index = 1:numel(solutions) % Loop over every solver output in the current trial.
        metrics = evaluate_solution(solver_data,solutions{method_index}.X,solutions{method_index}.Y,reference); % Evaluate the final solution only after the solver has terminated.
        method_result = build_method_result(solutions{method_index},metrics); % Combine solver statistics with independent evaluation metrics.
        row_counter = row_counter+1; % Advance the raw-result row counter.
        new_row = enrich_result(method_result,sweep_name,n,r,matrix_type); % Build one internal result row containing only grouping fields and requested metrics.
        new_row.trial = trial;
        new_row.problem_seed = problem_seed;
        new_row.initialization_seed = initialization_seed;
        if isfield(solutions{method_index},'gamma_next')
            new_row.final_gamma_next=solutions{method_index}.gamma_next;
            new_row.stop_reason=string(solutions{method_index}.stop_reason);
        elseif solutions{method_index}.iterations==0
            new_row.stop_reason=string('direct_solve');
        elseif solutions{method_index}.iterations<opt.max_iter
            new_row.stop_reason=string('original_stopping_rule');
        else
            new_row.stop_reason=string('original_stop_or_limit');
        end
        raw_rows(row_counter,1) = new_row; %#ok<AGROW> % Append the result row with an identical structure layout.
    end % Finish the method-result storage loop.
    if opt.save_histories % Check whether complete iteration histories should be saved.
        if opt.include_projection
            writetable(history1,fullfile(condition_directory,sprintf('trial_%02d_SDP_Projection.csv',trial)));
        end
        writetable(history2,fullfile(condition_directory,sprintf('trial_%02d_sBMF_GD.csv',trial))); % Save the sBMF-GD history.
        writetable(history3,fullfile(condition_directory,sprintf('trial_%02d_sBMF_GN.csv',trial))); % Save the sBMF-GN history.
        writetable(history4,fullfile(condition_directory,sprintf('trial_%02d_aBMF_our.csv',trial))); % Save the aBMF-our history.
        writetable(history5,fullfile(condition_directory,sprintf('trial_%02d_aBMF_adp.csv',trial))); % Save the aBMF-adp history.
    end % Finish the optional history-saving block.
    writetable(struct2table(raw_rows),fullfile(opt.output_dir,'all_trial_results.csv')); % Checkpoint completed trials, outside solver timers.
    fprintf('Completed %s, n=%d, rank=%d, repetition=%d/%d.\n',matrix_type,n,r,trial,opt.num_trials); % Report progress without displaying removed comparison fields.
    clear observed solver_data reference solutions solution1 solution2 solution3 solution4 solution5; % Release the current trial data before the next trial.
end % Finish the trial loop.
end % Finish the condition runner.

function observed = make_observed_data(n,r,matrix_type,requested_mode,explicit_threshold) % Generate an observed matrix representation while keeping the planted eigensystem local.
p = min(n,max([r+10,4*r,50])); % Choose the dimension of the informative spectral subspace.
[Q,~] = qr(randn(n,p),0); % Generate a local orthonormal basis used only to synthesize A.
top_values = linspace(12,5,r); % Define the leading positive eigenvalues of the synthetic matrix.
positive_tail_count = min(max(r,5),p-r); % Choose the initial number of additional positive eigenvalues.
if strcmpi(matrix_type,'Indefinite')&&p==n % Check whether a full informative basis must reserve a negative direction.
    positive_tail_count = min(positive_tail_count,max(p-r-1,0)); % Reserve at least one informative negative eigenvalue when no background subspace exists.
end % Finish the indefinite full-basis safeguard.
remaining_count = p-r-positive_tail_count; % Compute the number of remaining informative eigenvalues.
positive_tail = linspace(3,0.5,positive_tail_count); % Define positive eigenvalues excluded by the target rank.
if strcmpi(matrix_type,'PSD') % Check whether a positive semidefinite matrix is requested.
    remaining_values = linspace(0.4,0.05,max(remaining_count,0)); % Define a nonnegative informative spectral tail.
    floor_value = 0.02; % Define the nonnegative repeated background eigenvalue.
else % Handle the indefinite matrix case.
    negative_values = -linspace(0.5,6,max(remaining_count,1)); % Define a substantial negative informative spectral tail.
    remaining_values = negative_values(1:max(remaining_count,0)); % Match the required informative-subspace length.
    floor_value = -0.25; % Define a negative repeated background eigenvalue.
end % Finish the matrix-type spectral branch.
planted_values = [top_values,positive_tail,remaining_values]; % Combine all locally planted informative eigenvalues.
planted_values = planted_values(1:p); % Enforce the exact informative-subspace length.
mode = resolve_data_mode(requested_mode,n,explicit_threshold); % Resolve automatic explicit or matrix-free storage.
observed = struct(); % Initialize the observed-data structure returned by the generator.
observed.n = n; % Store the public matrix dimension.
observed.mode = mode; % Store the public observed-data representation mode.
observed.matrix_type = matrix_type; % Store the public matrix-type label.
if strcmpi(mode,'explicit') % Check whether an explicit dense matrix is requested.
    A = floor_value*eye(n)+Q*diag(planted_values-floor_value)*Q'; % Construct the observed symmetric data matrix A.
    A = 0.5*(A+A'); % Remove numerical asymmetry introduced by floating-point arithmetic.
    observed.A = A; % Store the explicit observed matrix as the only matrix-valued solver input.
    observed.apply = @(X)A*X; % Define matrix-times-block multiplication using only the explicit observed matrix.
    observed.fro_sq = sum(A(:).^2); % Compute the squared Frobenius norm directly from the observed matrix.
else % Handle the matrix-free representation for larger dimensions.
    observed.apply = build_matrix_free_oracle(Q,planted_values(:),floor_value); % Build an opaque matrix-times-block oracle without returning Q or the planted eigenvalues.
    observed.fro_sq = sum(planted_values.^2)+(n-p)*floor_value^2; % Store the exact observable Frobenius-norm scalar needed by the quadratic objective.
end % Finish the observed-data representation branch.
observed.spectral_bound = estimate_spectral_norm(observed.apply,n); % Estimate a public spectral scale using only matrix-vector products.
Q = []; %#ok<NASGU> % Remove the local planted eigenvector basis from the generator workspace after building A or the oracle.
planted_values = []; %#ok<NASGU> % Remove the local planted eigenvalues from the generator workspace after building A or the oracle.
end % Finish the leakage-free observed-data generator.

function mode = resolve_data_mode(requested_mode,n,explicit_threshold) % Resolve the requested observed-data representation mode.
if strcmpi(requested_mode,'auto') % Check whether automatic mode selection was requested.
    if n<=explicit_threshold % Check whether the matrix dimension is small enough for dense storage.
        mode = 'explicit'; % Use a plainly observable dense matrix for moderate dimensions.
    else % Handle larger matrix dimensions.
        mode = 'matrix_free'; % Use a matrix-times-block oracle to avoid dense memory growth.
    end % Finish the automatic size branch.
else % Handle an explicitly requested mode.
    mode = lower(char(requested_mode)); % Normalize the requested mode label.
end % Finish the data-mode resolution branch.
end % Finish the data-mode resolver.

function apply_function = build_matrix_free_oracle(Q,values,floor_value) % Create a matrix-times-block function that hides the local synthesis factors.
apply_function = @(X)floor_value*X+Q*bsxfun(@times,values-floor_value,Q'*X); % Return only the action of the observed matrix on a block with old-MATLAB compatibility.
end % Finish the matrix-free oracle builder.

function solver_data = make_solver_data(observed) % Build the restricted data structure passed to every solver.
solver_data = struct(); % Initialize the restricted solver-data structure.
solver_data.n = observed.n; % Expose the public matrix dimension.
solver_data.mode = observed.mode; % Expose the public data representation mode.
solver_data.matrix_type = observed.matrix_type; % Expose the public matrix-type label.
solver_data.apply = observed.apply; % Expose only multiplication by the observed matrix.
solver_data.fro_sq = observed.fro_sq; % Expose the matrix Frobenius-norm scalar required for objective evaluation.
solver_data.spectral_bound = observed.spectral_bound; % Expose a matrix-vector-product spectral-scale estimate.
if strcmpi(observed.mode,'explicit') % Check whether the observed matrix is explicitly stored.
    solver_data.A = observed.A; % Expose the observed matrix A itself and no synthesis factors.
end % Finish the explicit-data interface branch.
end % Finish the restricted solver-interface builder.

function verify_solver_interface(solver_data) % Verify that no planted spectral information is exposed to solvers.
forbidden_fields = {'Q','eigenvalues','planted_values','Ustar','star_values','reference'}; % Define fields that must never appear in solver input.
for field_index = 1:numel(forbidden_fields) % Loop over all forbidden field names.
    if isfield(solver_data,forbidden_fields{field_index}) % Check whether a forbidden field is present.
        error('Solver interface exposes forbidden spectral information: %s.',forbidden_fields{field_index}); % Stop when spectral leakage is detected.
    end % Finish the forbidden-field test.
end % Finish the forbidden-field validation loop.
end % Finish the solver-interface verifier.

function estimate = estimate_spectral_norm(apply_function,n) % Estimate the spectral norm using only matrix-vector products.
state = rng; % Preserve the caller random-number-generator state.
rng(13579+n,'twister'); % Set a deterministic local seed for the power iteration.
vector = randn(n,1); % Generate a random starting vector.
vector = vector/max(norm(vector),1e-30); % Normalize the starting vector.
estimate = 0; % Initialize the spectral-norm estimate.
for iteration = 1:30 % Run a fixed number of symmetric power iterations.
    product = apply_function(vector); % Apply the observed matrix to the current vector.
    new_estimate = norm(product); % Estimate the current operator magnitude.
    if new_estimate<=1e-30 % Check whether the observed matrix is numerically zero.
        estimate = 0; % Record a zero spectral estimate.
        break; % Exit the power iteration for the zero matrix.
    end % Finish the zero-matrix test.
    vector = product/new_estimate; % Normalize the next power-iteration vector.
    estimate = new_estimate; % Store the latest spectral-norm estimate.
end % Finish the power-iteration loop.
rng(state); % Restore the caller random-number-generator state.
estimate = max(1.1*estimate,1e-12); % Add a modest safety factor for initial step-size selection.
end % Finish the matrix-free spectral-norm estimator.

function reference = compute_reference(solver_data,r,tolerance,max_iterations,seed) % Recompute the best rank-r PSD reference from observed data only.
[U,lambda] = compute_top_eigenpairs(solver_data,r,tolerance,max_iterations,seed); % Compute leading eigenpairs without using planted information.
[lambda,order] = sort(real(lambda(:)),'descend'); % Sort the computed eigenvalues in descending algebraic order.
U = real(U(:,order)); % Reorder and realify the corresponding eigenvectors.
positive_lambda = max(lambda,0); % Project the leading eigenvalues onto the nonnegative orthant.
positive_mask = positive_lambda>max(tolerance,1e-14); % Identify numerically positive retained eigenvalues.
Upositive = U(:,positive_mask); % Retain the positive reference eigenvectors.
lambda_positive = positive_lambda(positive_mask); % Retain the positive reference eigenvalues.
reference.U = Upositive; % Store the independently computed positive reference eigenspace.
reference.lambda = lambda_positive; % Store the independently computed positive reference eigenvalues.
reference.factor = Upositive*diag(sqrt(lambda_positive)); % Store a factor of the independent rank-r PSD projection.
reference.fstar = 0.5*max(solver_data.fro_sq-sum(lambda_positive.^2),0); % Compute the reference objective from observed-data quantities.
end % Finish the independent reference computation.

function [U,lambda] = compute_top_eigenpairs(solver_data,r,tolerance,max_iterations,seed) % Compute leading eigenpairs from explicit A or the matrix-free oracle.
%options.issym = true; % Tell eigs that the observed operator is symmetric.
options.isreal = true; % Tell eigs that the observed operator is real.
options.tol = tolerance; % Set the eigensolver tolerance.
options.maxit = max_iterations; % Set the eigensolver iteration limit.
state = rng; % Preserve the caller random-number-generator state.
rng(seed,'twister'); % Set a deterministic local eigensolver seed.
initial_vector = randn(solver_data.n,1); % Generate a deterministic starting vector for eigs.
initial_vector = initial_vector/max(norm(initial_vector),1e-30); % Normalize the eigensolver starting vector.
options.v0 = initial_vector; % Supply the deterministic starting vector to eigs.
if strcmpi(solver_data.mode,'explicit') % Check whether the observed matrix is explicitly available.
    [U,D] = eigs(solver_data.A,r,'la',options); % Compute the algebraically largest eigenpairs from the observed matrix A without retaining a status metric.
else % Handle the matrix-free observed-data representation.
    [U,D] = eigs(solver_data.apply,solver_data.n,r,'la',options); % Compute the algebraically largest eigenpairs from the A-times-vector oracle without retaining a status metric.
end % Finish the observed-data eigensolver branch.
rng(state); % Restore the caller random-number-generator state.
lambda = diag(D); % Extract the computed eigenvalues.
end % Finish the observed-data leading-eigenpair computation.

function f = data_objective(solver_data,X,Y) % Evaluate one-half of the squared Frobenius residual from observed data.
XtX = X'*X; % Compute the Gram matrix of X.
YtY = Y'*Y; % Compute the Gram matrix of Y.
product_norm_sq = sum(sum(XtX.*YtY')); % Compute the squared Frobenius norm of X times Y transpose.
cross_term = sum(sum(X.*solver_data.apply(Y))); % Compute the inner product between A and X times Y transpose.
f = max(0,0.5*(solver_data.fro_sq+product_norm_sq-2*cross_term)); % Assemble the nonnegative quadratic approximation objective.
end % Finish the data-objective function.

function f = symmetric_objective(solver_data,X) % Evaluate the symmetric BMF objective from observed data.
f = data_objective(solver_data,X,X); % Reuse the general factorized objective with identical factors.
end % Finish the symmetric-objective function.

function value = residual_fro(solver_data,X,Y) % Evaluate the Frobenius norm of A minus X times Y transpose.
value = sqrt(max(0,2*data_objective(solver_data,X,Y))); % Convert the half-squared objective to a residual norm.
end % Finish the residual-norm function.

function [solution,history] = sdp_projection_solver(solver_data,r,tolerance,max_iterations,seed) % Compute the rank-r PSD projection from observed data only.
start_time = tic; % Start the projection wall-clock timer.
[U,lambda] = compute_top_eigenpairs(solver_data,r,tolerance,max_iterations,seed); % Independently compute leading eigenpairs from observed data.
[lambda,order] = sort(real(lambda(:)),'descend'); % Sort the computed eigenvalues in descending order.
U = real(U(:,order)); % Reorder and realify the computed eigenvectors.
lambda = max(lambda,0); % Project the retained eigenvalues onto the nonnegative orthant.
X = U*diag(sqrt(lambda)); % Build a factor of the observed-data rank-r PSD projection.
elapsed = toc(start_time); % Measure the projection wall-clock time.
data_value = symmetric_objective(solver_data,X); % Evaluate the projection data objective without any reference input.
solution = solver_solution('SDP-Projection',X,X,0,elapsed,NaN); % Build the projection solver output without exposing eigensolver status as a comparison metric.
history = array2table([0,elapsed,NaN,data_value,data_value,0,0,NaN],'VariableNames',history_names()); % Build a one-row projection history without reference-dependent metrics.
end % Finish the SDP-projection solver.

function [solution,history] = symmetric_gd_solver(solver_data,X0,max_iter,tol) % Run safeguarded gradient descent on the symmetric BMF objective.
X = X0; % Initialize the symmetric factor.
old_objective = symmetric_objective(solver_data,X); % Evaluate the initial objective.
records = nan(max_iter,8); % Preallocate the reference-free iteration-history array.
start_time = tic; % Start the wall-clock timer.
last_iteration = max_iter; % Initialize the reported final iteration.
for iteration = 1:max_iter % Start the gradient-descent loop.
    gradient = 2*(X*(X'*X)-solver_data.apply(X)); % Compute the symmetric BMF gradient from observed data.
    step = 1/max(4*(solver_data.spectral_bound+3*norm(X,2)^2),1e-12); % Compute a conservative initial step size.
    accepted = false; % Initialize the line-search acceptance flag.
    for line_search = 1:30 % Start a bounded backtracking line search.
        candidate = X-step*gradient; % Form the gradient-descent trial point.
        candidate_objective = symmetric_objective(solver_data,candidate); % Evaluate the trial objective from observed data.
        if candidate_objective<=old_objective-1e-4*step*norm(gradient,'fro')^2 % Test the Armijo sufficient-decrease condition.
            accepted = true; % Mark the trial point as accepted.
            break; % Exit the line-search loop.
        end % Finish the sufficient-decrease test.
        step = 0.5*step; % Reduce the step size after rejection.
    end % Finish the backtracking line search.
    if ~accepted % Check whether all line-search trials failed.
        candidate = X-step*gradient; % Use the final safeguarded gradient trial.
        candidate_objective = symmetric_objective(solver_data,candidate); % Evaluate the final trial objective.
    end % Finish the failed-line-search safeguard.
    X = candidate; % Accept the selected new iterate.
    elapsed = toc(start_time); % Measure the elapsed wall-clock time.
    relative_change = abs(candidate_objective-old_objective)/max(1,abs(old_objective)); % Compute the relative objective change.
    records(iteration,:) = [iteration,elapsed,NaN,candidate_objective,candidate_objective,0,0,step]; % Record reference-free solver diagnostics.
    old_objective = candidate_objective; % Update the previous objective value.
    if relative_change<tol % Test the objective-based stopping condition.
        last_iteration = iteration; % Record the stopping iteration.
        break; % Exit the gradient-descent loop.
    end % Finish the stopping test.
end % Finish the gradient-descent loop.
records = records(1:last_iteration,:); % Remove unused preallocated history rows.
history = array2table(records,'VariableNames',history_names()); % Convert the history array to a table.
solution = solver_solution('sBMF-GD',X,X,last_iteration,records(end,2),NaN); % Build the symmetric gradient-descent solver output.
end % Finish the symmetric gradient-descent solver.

function [solution,history] = symmetric_gauss_newton_solver(solver_data,X0,max_iter,tol,rank_floor) % Run the Liu-Wen-Zhang Gauss-Newton method using observed data only.
X = repair_rank(X0,rank_floor); % Ensure that the initial factor has full column rank.
old_objective = symmetric_objective(solver_data,X); % Evaluate the initial symmetric objective.
records = nan(max_iter,8); % Preallocate the reference-free Gauss-Newton history.
start_time = tic; % Start the Gauss-Newton wall-clock timer.
last_iteration = max_iter; % Initialize the reported final iteration.
for iteration = 1:max_iter % Start the Gauss-Newton iteration loop.
    gram = X'*X; % Form the factor Gram matrix.
    gram = 0.5*(gram+gram')+rank_floor*eye(size(gram)); % Symmetrize and regularize the Gram matrix.
    target = solver_data.apply(X)/gram-X; % Form the central Gauss-Newton target component from observed data.
    direction = target-0.5*X*(gram\(X'*target)); % Apply the efficient Gauss-Newton projection formula.
    step = 1; % Initialize the line-search step at the full Gauss-Newton step.
    accepted = false; % Initialize the line-search acceptance flag.
    gradient_for_derivative = 2*(X*(X'*X)-solver_data.apply(X)); % Compute the gradient used in the directional derivative.
    directional_derivative = sum(gradient_for_derivative(:).*direction(:)); % Compute the objective directional derivative.
    for line_search = 1:30 % Start a bounded backtracking line search.
        candidate = repair_rank(X+step*direction,rank_floor); % Form and rank-repair the trial factor.
        candidate_objective = symmetric_objective(solver_data,candidate); % Evaluate the trial objective from observed data.
        if candidate_objective<=old_objective+1e-4*step*directional_derivative % Test the Armijo condition for the Gauss-Newton direction.
            accepted = true; % Mark the trial step as accepted.
            break; % Exit the line-search loop.
        end % Finish the Gauss-Newton sufficient-decrease test.
        step = 0.5*step; % Reduce the trial step after rejection.
    end % Finish the Gauss-Newton line search.
    if ~accepted % Check whether the Gauss-Newton line search failed.
        gradient = 2*(X*(X'*X)-solver_data.apply(X)); % Compute a fallback gradient direction from observed data.
        step = 1/max(4*(solver_data.spectral_bound+3*norm(X,2)^2),1e-12); % Compute a conservative fallback gradient step.
        candidate = repair_rank(X-step*gradient,rank_floor); % Form and rank-repair the fallback trial factor.
        candidate_objective = symmetric_objective(solver_data,candidate); % Evaluate the fallback objective.
    end % Finish the Gauss-Newton fallback branch.
    X = candidate; % Accept the selected new factor.
    elapsed = toc(start_time); % Measure the elapsed Gauss-Newton time.
    relative_change = abs(candidate_objective-old_objective)/max(1,abs(old_objective)); % Compute the relative objective change.
    records(iteration,:) = [iteration,elapsed,NaN,candidate_objective,candidate_objective,0,0,step]; % Record reference-free solver diagnostics.
    old_objective = candidate_objective; % Update the previous objective.
    if relative_change<tol % Test the Gauss-Newton stopping condition.
        last_iteration = iteration; % Record the stopping iteration.
        break; % Exit the Gauss-Newton loop.
    end % Finish the stopping test.
end % Finish the Gauss-Newton loop.
records = records(1:last_iteration,:); % Remove unused history rows.
history = array2table(records,'VariableNames',history_names()); % Convert the Gauss-Newton history to a table.
solution = solver_solution('sBMF-GN',X,X,last_iteration,records(end,2),NaN); % Build the Gauss-Newton solver output.
end % Finish the symmetric Gauss-Newton solver.

function X = repair_rank(X,rank_floor) % Repair numerical column-rank deficiency by flooring singular values.
[U,S,V] = svd(X,'econ'); % Compute a compact singular value decomposition.
singular_values = diag(S); % Extract the singular values.
singular_values = max(singular_values,rank_floor); % Floor every singular value at the numerical safeguard.
X = U*diag(singular_values)*V'; % Reconstruct a full-column-rank factor.
end % Finish the rank-repair function.

function gamma = gamma_adp(previous,X,Y,gamma_max) % Evaluate the multiplicative adaptive penalty from the comparison literature.
numerator = sum(X(:).^2)+sum(Y(:).^2); % Compute the sum of squared factor norms.
denominator = 2*abs(sum(X(:).*Y(:))); % Compute twice the absolute factor inner product.
ratio = numerator/max(denominator,1e-30); % Form the safeguarded multiplicative ratio.
gamma = min(max(previous*ratio,1e-12),gamma_max); % Update and clamp the adaptive penalty.
end % Finish the comparison adaptive penalty update.

function [solution,history] = alternating_solver(name,update_mode,solver_data,X0,max_iter,tol,gamma0,epsilon,gamma_max) % Original adp solver.
assert(strcmpi(update_mode,'adp'),'Use mixed_alternating_solver for our.');
X = X0; % Initialize the first asymmetric factor.
Y = X0; % Initialize the second asymmetric factor identically.
gamma = gamma0; % Initialize the agreement penalty.
old_objective = data_objective(solver_data,X,Y); % Evaluate the initial data objective.
records = nan(max_iter,8); % Preallocate the reference-free alternating history.
start_time = tic; % Start the wall-clock timer.
last_iteration = max_iter; % Initialize the reported final iteration.
for iteration = 1:max_iter % Start the alternating-minimization loop.
    Ygram = Y'*Y+gamma*eye(size(Y,2)); % Form the regularized Y-block normal matrix.
    X = safe_right_solve(solver_data.apply(Y)+gamma*Y,Ygram); % Compute the exact X-block minimizer from observed data.
    Xgram = X'*X+gamma*eye(size(X,2)); % Form the regularized X-block normal matrix.
    Y = safe_right_solve(solver_data.apply(X)+gamma*X,Xgram); % Compute the exact Y-block minimizer from observed data.
    elapsed = toc(start_time); % Measure elapsed wall-clock time.
    data_value = data_objective(solver_data,X,Y); % Evaluate the current data objective.
    disagreement = norm(X-Y,'fro'); % Evaluate the current factor disagreement.
    relative_disagreement = disagreement/max([1,norm(X,'fro'),norm(Y,'fro')]); % Normalize the current factor disagreement.
    full_objective = data_value+0.5*gamma*disagreement^2; % Evaluate the complete asymmetric objective.
    records(iteration,:) = [iteration,elapsed,gamma,full_objective,data_value,disagreement,relative_disagreement,NaN]; % Record reference-free solver diagnostics.
    relative_change = abs(data_value-old_objective)/max(1,abs(old_objective)); % Compute the relative data-objective change.
    old_objective = data_value; % Update the previous data objective.
    gamma = gamma_adp(gamma,X,Y,gamma_max); % Original adp update; this solver is used only for adp.
    if relative_change<tol&&relative_disagreement<sqrt(tol) % Test the joint stopping condition.
        last_iteration = iteration; % Record the stopping iteration.
        break; % Exit the alternating-minimization loop.
    end % Finish the stopping test.
end % Finish the alternating-minimization loop.
records = records(1:last_iteration,:); % Remove unused history rows.
history = array2table(records,'VariableNames',history_names()); % Convert the alternating history to a table.
solution = solver_solution(name,X,Y,last_iteration,records(end,2),gamma); % Build the asymmetric solver output.
end % Finish the alternating-minimization solver.

function [solution,history] = mixed_alternating_solver(data,X0,max_iter,tol,gamma0,settings)
% X_k,Y_k are obtained with gamma_{k-1}; then gamma_mix_topk produces gamma_k.
X=X0;Y=X0;gamma=gamma0;
AY=data.apply(Y);GY=Y'*Y;
old_objective=0.5*max(data.fro_sq+sum(sum(GY.*GY))-2*sum(sum(X.*AY)),0);
names=[history_names(),{'gamma_previous','gamma_next','gamma_hat','gamma_check', ...
    'directional_quotient','directional_numerator','D_norm_sq','loss_gradient_fro', ...
    'D_zero','operator_fallback','relative_objective_change'}];
records=nan(max_iter,numel(names));stop_reason='iteration_limit';
clock_start=tic;
for k=1:max_iter
    used=gamma;
    X=safe_right_solve(AY+used*Y,GY+used*eye(size(Y,2)));
    AX=data.apply(X);GX=X'*X;
    Y=safe_right_solve(AX+used*X,GX+used*eye(size(X,2)));
    AY=data.apply(Y);GY=Y'*Y;
    residual_sq=max(data.fro_sq+sum(sum(GX.*GY))-2*sum(sum(X.*AY)),0);
    data_value=0.5*residual_sq;
    cache=struct('AX',AX,'AY',AY,'loss_fro',sqrt(residual_sq));
    [next,info]=gamma_mix_topk(used,X,Y,data,settings,cache);
    disagreement=norm(X-Y,'fro');
    relative_disagreement=disagreement/max([1,norm(X,'fro'),norm(Y,'fro')]);
    relative_change=abs(data_value-old_objective)/max(1,abs(old_objective));
    full_objective=data_value+0.5*used*disagreement^2;
    records(k,:)=[k,toc(clock_start),used,full_objective,data_value,disagreement,relative_disagreement,NaN, ...
        used,next,info.gamma_hat,info.gamma_check,info.directional_quotient, ...
        info.directional_numerator,info.D_norm_sq,info.loss_gradient_fro, ...
        double(info.D_zero),double(info.operator_fallback),relative_change];
    if k==1||mod(k,50)==0
        fprintf('  our k=%d: gamma_used=%.6g, gamma_next=%.6g, relD=%.3g, df=%.3g\n', ...
            k,used,next,relative_disagreement,relative_change);
    end
    old_objective=data_value;gamma=next;
    if relative_change<tol&&relative_disagreement<sqrt(tol)
        stop_reason='joint_tolerance';break;
    end
end
history=array2table(records(1:k,:),'VariableNames',names);
solution=solver_solution('aBMF-our',X,Y,k,history.time(end),history.gamma(end));
solution.gamma_next=gamma;solution.stop_reason=stop_reason;
end

function X = safe_right_solve(right_hand_side,matrix) % Solve a small right-side linear system with conditioning safeguards.
matrix = 0.5*(matrix+matrix'); % Symmetrize the small normal matrix.
if rcond(matrix)<1e-12 % Check whether the normal matrix is poorly conditioned.
    matrix = matrix+1e-10*eye(size(matrix)); % Add a small diagonal regularization.
end % Finish the conditioning safeguard.
X = right_hand_side/matrix; % Solve the right-side linear system.
end % Finish the safeguarded right solve.

function solution = solver_solution(name,X,Y,iterations,time,final_gamma) % Build a solver output containing only factors and requested solver statistics.
solution.method = string(name); % Store the method name.
solution.X = X; % Store the final first factor.
solution.Y = Y; % Store the final second factor.
solution.iterations = iterations; % Store the solver iteration count.
solution.time = time; % Store the solver wall-clock time.
solution.final_gamma = final_gamma; % Store the final asymmetric penalty when applicable.
end % Finish the reference-free solver-output builder.

function metrics = evaluate_solution(solver_data,X,Y,reference) % Evaluate every completed solver output with the same original PSD objective and subspace metric.
common_factor = 0.5*(X+Y); % Form the common symmetric factor, which equals X for SDP and sBMF because those methods return X equal to Y.
f = symmetric_objective(solver_data,common_factor); % Evaluate one-half of the squared residual of A minus common_factor times common_factor transpose for every method.
metrics.data_objective = f; % Store the unified original low-rank PSD objective value.
metrics.rel_objective = abs(f-reference.fstar)/max(1,abs(reference.fstar)); % Store the unified relative objective gap to the independently recomputed optimum.
metrics.subspace_error = subspace_error(common_factor,reference.U); % Store the unified Frobenius sine-principal-angle error of the common factor.
metrics.rel_disagreement = norm(X-Y,'fro')/max([1,norm(X,'fro'),norm(Y,'fro')]); % Store the unified final relative disagreement, which is zero for SDP and sBMF outputs.
end % Finish the unified final-solution evaluator.

function value = subspace_error(Z,Ustar) % Evaluate the Frobenius sine-principal-angle error with a rank-deficiency safeguard.
if isempty(Ustar) % Check whether the independent reference has no positive eigenvectors.
    value = 0; % Return zero for an empty target subspace.
    return; % Exit the empty-reference branch.
end % Finish the empty-reference test.
[Uz,Sz,~] = svd(Z,'econ'); % Compute an orthonormal singular-vector basis for the estimated common factor.
singular_values = diag(Sz); % Extract the singular values used to determine the numerical factor rank.
if isempty(singular_values) % Check whether the estimated factor contains no singular directions.
    estimated_rank = 0; % Set the numerical estimated rank to zero.
else % Handle a nonempty singular-value vector.
    rank_threshold = max(size(Z))*eps(max(singular_values)); % Define a standard scale-aware numerical-rank threshold.
    estimated_rank = sum(singular_values>rank_threshold); % Count the numerically nonzero estimated factor directions.
end % Finish the numerical-rank branch.
if estimated_rank==0 % Check whether the estimated common factor is numerically zero.
    overlap = 0; % Assign zero overlap with the target subspace.
else % Handle a nonzero estimated factor subspace.
    Qz = Uz(:,1:estimated_rank); % Retain only the numerically identifiable estimated subspace directions.
    overlap = norm(Qz'*Ustar,'fro')^2; % Compute the squared overlap between the estimated and reference subspaces.
end % Finish the estimated-subspace branch.
r = size(Ustar,2); % Read the independent target subspace dimension.
value = sqrt(max(0,r-min(r,overlap))); % Return the Frobenius norm of the sine of the principal-angle vector.
end % Finish the rank-aware subspace-error function.

function result = build_method_result(solution,metrics) % Combine solver statistics with the three unified independently evaluated comparison metrics.
result.method = solution.method; % Store the method name.
result.iterations = solution.iterations; % Store the number of outer solver iterations, with zero used for the direct SDP projection.
result.time = solution.time; % Store only the solver wall-clock time and exclude the independent reference computation.
result.final_gamma = solution.final_gamma; % Store the final asymmetric penalty and retain NaN for methods without a gamma parameter.
result.data_objective = metrics.data_objective; % Store the unified original low-rank PSD objective.
result.rel_objective = metrics.rel_objective; % Store the unified relative objective gap.
result.subspace_error = metrics.subspace_error; % Store the unified subspace error.
result.rel_disagreement = metrics.rel_disagreement; % Store the unified final relative factor disagreement.
end % Finish the method-result builder.

function enriched = enrich_result(result,sweep_name,n,r,matrix_type) % Add only condition keys and requested metrics to one internal result row.
enriched = result_row_template(); % Start from the common field template to preserve identical fields and order.
enriched.sweep = string(sweep_name); % Store the internal sweep identifier used only to separate the two aggregate tables.
enriched.matrix_type = string(matrix_type); % Store whether the observed matrix is PSD or indefinite.
enriched.n = n; % Store the matrix dimension.
enriched.rank = r; % Store the target rank.
enriched.method = result.method; % Store the method name.
enriched.iterations = result.iterations; % Store the iteration count for internal aggregation.
enriched.time = result.time; % Store the wall-clock time for internal aggregation.
enriched.final_gamma = result.final_gamma; % Store the final aBMF penalty when applicable.
enriched.data_objective = result.data_objective; % Store the unified final low-rank PSD objective.
enriched.rel_objective = result.rel_objective; % Store the unified relative objective gap.
enriched.subspace_error = result.subspace_error; % Store the unified reference-subspace error.
enriched.rel_disagreement = result.rel_disagreement; % Store the unified final relative factor disagreement.
end % Finish the reduced result-enrichment function.

function row = result_row_template() % Create one internal result-row structure with only grouping fields and requested metrics.
row = struct( ... % Construct the common scalar structure used by every internal result row.
    'sweep',string(''), ... % Reserve the internal sweep identifier used only before output selection.
    'matrix_type',string(''), ... % Reserve the matrix-type field.
    'n',NaN, ... % Reserve the matrix-dimension field.
    'rank',NaN, ... % Reserve the target-rank field.
    'method',string(''), ... % Reserve the method-name field.
    'trial',NaN, ...
    'problem_seed',NaN, ...
    'initialization_seed',NaN, ...
    'final_gamma_next',NaN, ...
    'stop_reason',string(''), ...
    'iterations',NaN, ... % Reserve the iteration-count field.
    'time',NaN, ... % Reserve the wall-clock-time field.
    'final_gamma',NaN, ... % Reserve the final-penalty field.
    'data_objective',NaN, ... % Reserve the unified data-objective field.
    'rel_objective',NaN, ... % Reserve the relative-objective-gap field.
    'subspace_error',NaN, ... % Reserve the subspace-error field.
    'rel_disagreement',NaN); % Reserve the relative-disagreement field and finish the common structure.
end % Finish the reduced fixed result-row template function.

function summary = aggregate_results(raw_table) % Aggregate every requested metric by sweep, matrix type, n, rank, and method.
keys = raw_table(:,{'sweep','matrix_type','n','rank','method'}); % Extract the grouping keys while keeping the sweep internal.
[groups,key_table] = findgroups(keys); % Identify unique experimental condition and method groups.
metric_names = {'iterations','time','final_gamma','data_objective','rel_objective','subspace_error','rel_disagreement'}; % Define the seven requested metrics.
summary = key_table; % Initialize the aggregate table with the grouping keys.
summary.num_trials = splitapply(@numel,raw_table.iterations,groups); % Count the number of repeated trials in each group without storing trial identifiers.
for metric_index = 1:numel(metric_names) % Loop over all requested metrics.
    metric_name = metric_names{metric_index}; % Read the current metric name.
    values = raw_table.(metric_name); % Extract the internal per-run values.
    summary.([metric_name,'_mean']) = splitapply(@finite_mean,values,groups); % Compute the mean over finite values.
    summary.([metric_name,'_std']) = splitapply(@finite_std,values,groups); % Compute the sample standard deviation over finite values.
end % Finish the metric aggregation loop.
summary = sortrows(summary,{'sweep','matrix_type','n','rank','method'}); % Sort the aggregate table deterministically.
end % Finish the aggregate-only result function.

function comparison = select_comparison_columns(summary) % Select only aggregate comparison columns in the requested order.
variable_names = {'matrix_type','n','rank','method','num_trials','iterations_mean','iterations_std','time_mean','time_std','final_gamma_mean','final_gamma_std','data_objective_mean','data_objective_std','rel_objective_mean','rel_objective_std','subspace_error_mean','subspace_error_std','rel_disagreement_mean','rel_disagreement_std'}; % Define the exact aggregate-only output layout.
comparison = summary(:,variable_names); % Return only the requested aggregate fields.
end % Finish the aggregate comparison-column selector.

function combined = combine_unique_summaries(first_summary,second_summary) % Combine two sweep summaries without duplicating an overlapping n-r condition.
combined = [first_summary;second_summary]; % Concatenate the two aggregate tables with identical columns.
keys = strcat(cellstr(combined.matrix_type),'|',cellstr(string(combined.n)),'|',cellstr(string(combined.rank)),'|',cellstr(combined.method)); % Build one stable textual key for each aggregate row.
[~,first_indices] = unique(keys,'stable'); % Retain the first occurrence of each matrix-type, n, rank, and method combination.
combined = combined(first_indices,:); % Remove only duplicate overlap rows between the two sweeps.
combined = sortrows(combined,{'matrix_type','n','rank','method'}); % Sort the combined aggregate table deterministically.
end % Finish the combined-summary deduplication function.

function value = finite_mean(values) % Compute a mean after discarding nonfinite entries.
values = values(isfinite(values)); % Retain only finite metric values.
if isempty(values) % Check whether any finite values remain.
    value = NaN; % Return NaN when the metric is unavailable for the group.
else % Handle a nonempty finite value set.
    value = mean(values); % Compute the ordinary arithmetic mean.
end % Finish the finite-mean branch.
end % Finish the finite-mean function.

function value = finite_std(values) % Compute a standard deviation after discarding nonfinite entries.
values = values(isfinite(values)); % Retain only finite metric values.
if isempty(values) % Check whether the metric is unavailable for the entire group.
    value = NaN; % Return NaN when no method-specific value exists, such as gamma for SDP and sBMF.
elseif numel(values)==1 % Check whether exactly one finite observation remains.
    value = 0; % Return zero because a single observation has no across-trial variation.
else % Handle at least two finite observations.
    value = std(values,0); % Compute the sample standard deviation.
end % Finish the finite-standard-deviation branch.
end % Finish the finite-standard-deviation function.

function gamma0 = resolve_gamma(value,auto_gamma) % Resolve the initial asymmetric penalty value.
if ischar(value) || isstring(value) % Check whether the penalty is supplied as text.
    if strcmpi(char(value),'auto') % Check whether automatic selection is requested.
        gamma0 = auto_gamma; % Use the automatically computed penalty.
    else % Handle a numeric value encoded as text.
        gamma0 = str2double(char(value)); % Convert the text to a numeric scalar.
    end % Finish the text-value branch.
else % Handle an already numeric penalty value.
    gamma0 = value; % Copy the numeric penalty value.
end % Finish the penalty-type branch.
if ~(isfinite(gamma0)&&gamma0>0) % Validate the resolved penalty value.
    error('gamma0 must be ''auto'' or a positive number.'); % Stop when the penalty is invalid.
end % Finish the penalty validation block.
end % Finish the penalty resolver.

function names = history_names() % Return the common reference-free history-table variable names.
names = {'iteration','time','gamma','full_objective','data_objective','disagreement','relative_disagreement','step'}; % Define the ordered solver-history columns.
end % Finish the history-name function.

function plot_sweep_metrics(summary,x_variable,output_directory) % Plot aggregate method comparisons for one sweep.
if ~exist(output_directory,'dir') % Check whether the figure output directory exists.
    mkdir(output_directory); % Create the figure output directory when necessary.
end % Finish the figure-directory creation block.
metric_names = {'data_objective','rel_objective','subspace_error','rel_disagreement','time','iterations','final_gamma'}; % Define exactly the requested aggregate metrics plotted for every sweep.
ylabels = {'Data objective','Relative objective gap','Subspace error','Relative disagreement','Wall-clock time (s)','Iterations','Final gamma'}; % Define readable vertical-axis labels for the requested metrics.
method_names = unique(summary.method,'stable'); % Retrieve all compared method names.
matrix_types = unique(summary.matrix_type,'stable'); % Retrieve PSD and indefinite matrix labels.
for type_index = 1:numel(matrix_types) % Loop over the matrix types.
    type_name = matrix_types(type_index); % Read the current matrix-type label.
    type_rows = summary(summary.matrix_type==type_name,:); % Select aggregate rows for the current matrix type.
    for metric_index = 1:numel(metric_names) % Loop over every major metric.
        metric_name = metric_names{metric_index}; % Read the current metric name.
        figure_handle = figure('Visible','off','Color','w'); % Create an invisible white-background figure.
        hold on; % Allow all methods to share the same axes.
        for method_index = 1:numel(method_names) % Loop over every compared method.
            method_name = method_names(method_index); % Read the current method name.
            rows = type_rows(type_rows.method==method_name,:); % Select rows for the current method.
            rows = sortrows(rows,x_variable); % Sort the rows by rank or matrix dimension.
            x_values = rows.(x_variable); % Extract the horizontal-axis values.
            mean_values = rows.([metric_name,'_mean']); % Extract the metric means.
            std_values = rows.([metric_name,'_std']); % Extract the metric standard deviations.
            valid = isfinite(mean_values); % Identify finite metric values.
            if ~any(valid) % Check whether this method has any available values.
                continue; % Skip methods without finite values for the current metric.
            end % Finish the unavailable-method test.
            errorbar(x_values(valid),max(mean_values(valid),1e-16),std_values(valid),'o-','LineWidth',1.2,'DisplayName',char(method_name)); % Plot means with standard-deviation error bars.
        end % Finish the method plotting loop.
        grid on; % Enable grid lines.
        box on; % Draw a box around the axes.
        xlabel(x_variable); % Label the horizontal axis as rank or n.
        ylabel(ylabels{metric_index}); % Label the vertical axis with the current metric.
        title(sprintf('%s: %s versus %s',char(type_name),ylabels{metric_index},x_variable)); % Add a descriptive plot title.
        legend('Location','best'); % Display the method legend.
        if any(strcmp(metric_name,{'data_objective','rel_objective','subspace_error','rel_disagreement','time','final_gamma'})) % Check whether logarithmic scaling improves readability.
            set(gca,'YScale','log'); % Use a logarithmic vertical scale for positive spread metrics.
        end % Finish the logarithmic-scale selection.
        file_name = sprintf('%s_%s_vs_%s.png',lower(char(type_name)),metric_name,x_variable); % Build the output figure filename.
        save_figure_compatibly(figure_handle,fullfile(output_directory,file_name)); % Save the figure with old-MATLAB compatibility.
        close(figure_handle); % Close the invisible figure.
    end % Finish the metric plotting loop.
end % Finish the matrix-type plotting loop.
end % Finish the sweep-plotting function.

function save_figure_compatibly(figure_handle,path) % Save a figure across old and new MATLAB releases.
try % Try the modern high-resolution export function.
    exportgraphics(figure_handle,path,'Resolution',180); % Export the figure as a high-resolution PNG file.
catch % Handle MATLAB releases without exportgraphics.
    set(figure_handle,'PaperPositionMode','auto'); % Preserve the on-screen figure size for printing.
    print(figure_handle,path,'-dpng','-r180'); % Save the figure with the legacy print function.
end % Finish the figure-export compatibility branch.
pdf_path=regexprep(path,'\.png$','.pdf');
try
    exportgraphics(figure_handle,pdf_path,'ContentType','vector');
catch
    set(figure_handle,'PaperPositionMode','auto');
    print(figure_handle,pdf_path,'-dpdf','-bestfit');
end
end % Finish the compatible figure-saving function.

function write_json(path,structure) % Save a MATLAB structure as JSON text across MATLAB versions.
text = jsonencode(structure); % Basic interface: no version-dependent optional name/value parameters.
file_identifier = fopen(path,'w'); % Open the target JSON file for writing.
if file_identifier<0 % Check whether the output file was opened successfully.
    error('Cannot write %s',path); % Stop when the configuration file cannot be created.
end % Finish the file-open validation.
cleanup_object = onCleanup(@()fclose(file_identifier)); %#ok<NASGU> % Guarantee that the file is closed on function exit.
fwrite(file_identifier,unicode2native(text,'UTF-8'),'uint8'); % Preserve Unicode paths/errors on Windows.
end % Finish the JSON writer.
