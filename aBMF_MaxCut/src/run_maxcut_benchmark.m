function summary = run_maxcut_benchmark(varargin) % Run the Max-Cut comparison designed for the NeurIPS rebuttal.
%RUN_MAXCUT_BENCHMARK Compare a full SDP reference, symmetric BMF solvers, and asymmetric BMF solvers. % Describe the experiment purpose.
% The standard weighted relaxation is max 0.25*<L,Z> subject to Z being PSD and diag(Z)=1, with signed weights allowed. % State the weighted Max-Cut SDP relaxation.
% Symmetric factors use unit rows, while aBMF enforces diag(X*Y')=1 through an explicit multiplier. % State the feasibility convention.
% The asymmetric methods retain the standard diagonal equality through an augmented-Lagrangian multiplier. % State the constrained asymmetric formulation.
% aBMF-our uses the latest directional/norm mixed rule for the effective ALM loss.
% Example condition_list = {'data/G_small',50;'data/G_large',50;'data/G_large',100}; % Show the compact condition-list format.
% Example summary = run_maxcut_benchmark('condition_list',condition_list,'num_trials',5); % Show a compact benchmark call.
ip = inputParser; % Create the experiment input parser.
addParameter(ip,'condition_list',cell(0,2)); % Add an optional exact list of graph-file and rank pairs.
addParameter(ip,'graph_files',{}); % Add graph files used with the Cartesian rank list when no exact list is supplied.
addParameter(ip,'ranks',50); % Add low-rank dimensions used with graph_files.
addParameter(ip,'synthetic_sizes',[300 1000]); % Add fallback synthetic graph sizes for a quick standalone test.
addParameter(ip,'synthetic_probability',0.02); % Add the edge probability used by fallback synthetic graphs.
addParameter(ip,'synthetic_seed',1); % Add the base seed used by fallback synthetic graphs.
addParameter(ip,'num_trials',5); % Add the number of independent low-rank optimization trials.
addParameter(ip,'rounds',100); % Add the number of shared random-hyperplane rounding rounds.
addParameter(ip,'max_iter',1000); % Add the maximum number of iterations or sweeps.
addParameter(ip,'tol',1e-6); % Add the relative stopping tolerance.
addParameter(ip,'epsilon',1e-6); % Add the positive lower safeguard for gamma.
addParameter(ip,'gamma0','auto'); % Add the common initial gamma specification.
addParameter(ip,'gamma_max',1e8); % Add the numerical upper safeguard for adaptive gamma values.
addParameter(ip,'rho0',1); % Add the initial augmented-Lagrangian penalty for diag(X*Y')=1.
addParameter(ip,'rho_growth',2); % Add the multiplier-penalty growth factor.
addParameter(ip,'rho_max',1e6); % Add the upper safeguard for the multiplier penalty.
addParameter(ip,'multiplier_update_period',1); % Add the number of factor sweeps between multiplier updates.
addParameter(ip,'constraint_tol',1e-6); % Add the diagonal-equality feasibility tolerance.
addParameter(ip,'residual_reduction',0.9); % Add the required residual reduction before rho is increased.
addParameter(ip,'gamma_base',0.25); % Add the normalized-Laplacian numerical baseline for gamma.
addParameter(ip,'sbmf_step',1); % Add the initial Riemannian projected-gradient step.
addParameter(ip,'backtrack_factor',0.5); % Add the projected-gradient backtracking reduction factor.
addParameter(ip,'armijo_constant',1e-4); % Add the projected-gradient Armijo constant.
addParameter(ip,'full_sdp',true); % Add the option to run the dense full-SDP projected-gradient reference.
addParameter(ip,'max_full_sdp_n',1200); % Add the largest graph size allowed for the dense full-SDP reference.
addParameter(ip,'sdp_max_iter',100); % Add the maximum number of full-SDP projected-gradient iterations.
addParameter(ip,'sdp_step',1); % Add the full-SDP projected-gradient step size.
addParameter(ip,'dykstra_max_iter',100); % Add the maximum Dykstra projection iterations.
addParameter(ip,'dykstra_tol',1e-8); % Add the Dykstra projection tolerance.
addParameter(ip,'require_nonnegative_weights',false); % Optionally reject signed edge weights; false supports the full weighted Gset benchmark.
addParameter(ip,'normalize_laplacian',true); % Normalize the Laplacian for optimization while reporting original-scale objectives.
addParameter(ip,'best_known_file',''); % Add an optional CSV file with graph_name and best_known_cut columns.
addParameter(ip,'save_histories',false); % Add the option to save per-iteration histories.
addParameter(ip,'verbose',true); % Add the option to print condition, method, and iteration progress.
addParameter(ip,'progress_period',100); % Add the number of iterations between progress messages.
addParameter(ip,'output_dir',fullfile(pwd,'maxcut_results')); % Add the result-output directory.
addParameter(ip,'gamma_nu',0.9);
addParameter(ip,'gamma_mix_epsilon0',1e-6);
addParameter(ip,'trial_ids',[]);
parse(ip,varargin{:}); % Parse all supplied experiment options.
opt = ip.Results;
if isempty(opt.trial_ids),opt.trial_ids=1:opt.num_trials;end
assert(all(opt.trial_ids>=1)&all(opt.trial_ids<=opt.num_trials)&all(opt.trial_ids==fix(opt.trial_ids)));
assert(opt.gamma_nu>0&&opt.gamma_nu<1&&opt.gamma_mix_epsilon0>0); % Store the parsed experiment options.
validate_options(opt); % Validate all scalar and list experiment settings.
if ~exist(opt.output_dir,'dir') % Check whether the output directory exists.
    mkdir(opt.output_dir); % Create the output directory when necessary.
end % Finish the output-directory branch.
conditions = build_conditions(opt); % Build the exact graph-and-rank conditions to run.
if opt.verbose % Check whether benchmark progress should be printed.
    fprintf('Starting Max-Cut benchmark with %d conditions, %d trials, and %d maximum sweeps per method.\n',size(conditions,1),opt.num_trials,opt.max_iter); % Print the benchmark workload.
    drawnow; % Flush the command-window progress message.
end % Finish the benchmark-overview branch.
best_known = load_best_known(opt.best_known_file); % Load optional best-known cut values.
raw_rows = repmat(result_row_template(),0,1); % Initialize the raw result array with a fixed field layout.
assignment_records = repmat(assignment_template(),0,1); % Initialize saved best-cut assignments.
row_counter = 0; % Initialize the raw result counter.
assignment_counter = 0; % Initialize the assignment result counter.
graph_cache = containers.Map('KeyType','char','ValueType','any'); % Cache graphs that appear at multiple ranks.
full_sdp_done = containers.Map('KeyType','char','ValueType','logical'); % Track graphs whose full-SDP reference has already run.
for condition_index = 1:size(conditions,1) % Loop over all requested graph-and-rank conditions.
    source = conditions{condition_index,1}; % Read the current graph source.
    rank_value = conditions{condition_index,2}; % Read the current low-rank dimension.
    cache_key = graph_cache_key(source); % Build a stable graph-cache key.
    if isKey(graph_cache,cache_key) % Check whether the graph has already been loaded.
        graph = graph_cache(cache_key); % Reuse the cached graph.
    else % Handle a graph that has not yet been loaded.
        graph = obtain_graph(source,opt,condition_index); % Load or generate the current graph.
        graph = prepare_graph(graph,opt); % Build Laplacians and normalized optimization operators.
        graph_cache(cache_key) = graph; % Cache the prepared graph.
    end % Finish the graph-cache branch.
    if rank_value>graph.n % Check whether the requested rank is valid.
        error('Rank %d exceeds n=%d for graph %s.',rank_value,graph.n,graph.name); % Stop on an invalid low-rank dimension.
    end % Finish the rank validation.
    if opt.verbose % Check whether condition progress should be printed.
        fprintf('\n[Condition %d/%d] graph=%s, n=%d, m=%d, rank=%d, signed=%d\n',condition_index,size(conditions,1),graph.name,graph.n,graph.m,rank_value,~graph.weights_nonnegative); % Print the current condition and whether negative edge weights are present.
        drawnow; % Flush the condition progress message.
    end % Finish the condition-progress branch.
    if opt.full_sdp && graph.n<=opt.max_full_sdp_n && ~isKey(full_sdp_done,cache_key) % Check whether the full-SDP reference should run once.
        full_rounding_seed = 700000+condition_index; % Define a deterministic rounding seed for the full-SDP reference.
        print_method_start(opt,'Full-SDP-PG',0,0); % Print the full-SDP start message.
        [full_result,full_history,full_assignment] = solve_full_sdp_pg(graph,opt,full_rounding_seed,best_known); % Run the dense full-SDP reference.
        print_method_done(opt,full_result); % Print the full-SDP completion message.
        row_counter = row_counter+1; % Advance the raw result counter.
        raw_rows(row_counter,1) = full_result; % Store the full-SDP result.
        assignment_counter = assignment_counter+1; % Advance the assignment counter.
        assignment_records(assignment_counter,1) = make_assignment_record(graph.name,0,'Full-SDP-PG',full_assignment); % Store the full-SDP rounded assignment.
        if opt.save_histories % Check whether the full-SDP history should be saved.
            history_name = safe_filename(sprintf('%s_rank0_Full-SDP-PG.csv',graph.name)); % Build a safe history filename.
            writetable(full_history,fullfile(opt.output_dir,history_name)); % Save the full-SDP history.
        end % Finish the full-SDP history branch.
        full_sdp_done(cache_key) = true; % Mark the full-SDP reference as completed for this graph.
    end % Finish the optional full-SDP branch.
    for trial = opt.trial_ids % Loop over independent low-rank trials.
        if opt.verbose % Check whether trial progress should be printed.
            fprintf('[Trial %d/%d] initialization and shared rounding seeds prepared.\n',trial,opt.num_trials); % Print the current trial.
            drawnow; % Flush the trial progress message.
        end % Finish the trial-progress branch.
        initialization_seed = 100000*condition_index+trial; % Define the common factor-initialization seed.
        rounding_seed = 200000*condition_index+trial; % Define the common random-hyperplane seed.
        rng(initialization_seed,'twister'); % Set the common factor-initialization random stream.
        X0 = normalize_rows(randn(graph.n,rank_value)); % Generate the shared feasible low-rank initialization.
        gamma_initial = resolve_gamma_alm(opt.gamma0,opt.gamma_base,graph.gradient_fro,opt.epsilon); % Resolve the shared initial gamma from the lifted-gradient scale.
        rng(rounding_seed,'twister'); % Set the common low-rank rounding random stream.
        shared_hyperplanes = randn(rank_value,opt.rounds); % Generate shared hyperplanes for all low-rank methods.
        print_method_start(opt,'sBMF-PGD',trial,opt.num_trials); % Print the projected-gradient start message.
        [method_result,method_history,assignment] = solve_sbmf_pgd(graph,X0,opt,shared_hyperplanes,best_known); % Run symmetric BMF projected gradient.
        print_method_done(opt,method_result); % Print the projected-gradient completion message.
        row_counter = row_counter+1; % Advance the raw result counter.
        raw_rows(row_counter,1) = method_result;
        raw_rows(row_counter).trial=trial;
        raw_rows(row_counter).initialization_seed=initialization_seed;
        raw_rows(row_counter).rounding_seed=rounding_seed; % Store the symmetric projected-gradient result.
        assignment_counter = assignment_counter+1; % Advance the assignment counter.
        assignment_records(assignment_counter,1) = make_assignment_record(graph.name,rank_value,'sBMF-PGD',assignment); % Store the rounded assignment.
        save_history_if_requested(method_history,graph.name,rank_value,'sBMF-PGD',trial,opt); % Save the optional projected-gradient history.
        print_method_start(opt,'sBMF-Mixing',trial,opt.num_trials); % Print the Mixing start message.
        [method_result,method_history,assignment] = solve_sbmf_mixing(graph,X0,opt,shared_hyperplanes,best_known); % Run the symmetric Mixing baseline.
        print_method_done(opt,method_result); % Print the Mixing completion message.
        row_counter = row_counter+1; % Advance the raw result counter.
        raw_rows(row_counter,1) = method_result;
        raw_rows(row_counter).trial=trial;
        raw_rows(row_counter).initialization_seed=initialization_seed;
        raw_rows(row_counter).rounding_seed=rounding_seed; % Store the symmetric Mixing result.
        assignment_counter = assignment_counter+1; % Advance the assignment counter.
        assignment_records(assignment_counter,1) = make_assignment_record(graph.name,rank_value,'sBMF-Mixing',assignment); % Store the rounded assignment.
        save_history_if_requested(method_history,graph.name,rank_value,'sBMF-Mixing',trial,opt); % Save the optional Mixing history.
        print_method_start(opt,'aBMF-our',trial,opt.num_trials); % Print the proposed asymmetric start message.
        [method_result,method_history,assignment] = solve_abmf_alm(graph,X0,opt,shared_hyperplanes,best_known,'our',gamma_initial); % Run the proposed constrained asymmetric ALM method.
        print_method_done(opt,method_result); % Print the proposed asymmetric completion message.
        row_counter = row_counter+1; % Advance the raw result counter.
        raw_rows(row_counter,1) = method_result;
        raw_rows(row_counter).trial=trial;
        raw_rows(row_counter).initialization_seed=initialization_seed;
        raw_rows(row_counter).rounding_seed=rounding_seed; % Store the proposed asymmetric result.
        assignment_counter = assignment_counter+1; % Advance the assignment counter.
        assignment_records(assignment_counter,1) = make_assignment_record(graph.name,rank_value,'aBMF-our',assignment); % Store the rounded assignment.
        save_history_if_requested(method_history,graph.name,rank_value,'aBMF-our',trial,opt); % Save the optional proposed-method history.
        print_method_start(opt,'aBMF-adp',trial,opt.num_trials); % Print the adaptive asymmetric start message.
        [method_result,method_history,assignment] = solve_abmf_alm(graph,X0,opt,shared_hyperplanes,best_known,'adp',gamma_initial); % Run the constrained asymmetric comparison method.
        print_method_done(opt,method_result); % Print the adaptive asymmetric completion message.
        row_counter = row_counter+1; % Advance the raw result counter.
        raw_rows(row_counter,1) = method_result;
        raw_rows(row_counter).trial=trial;
        raw_rows(row_counter).initialization_seed=initialization_seed;
        raw_rows(row_counter).rounding_seed=rounding_seed; % Store the literature adaptive result.
        assignment_counter = assignment_counter+1; % Advance the assignment counter.
        assignment_records(assignment_counter,1) = make_assignment_record(graph.name,rank_value,'aBMF-adp',assignment); % Store the rounded assignment.
        save_history_if_requested(method_history,graph.name,rank_value,'aBMF-adp',trial,opt); % Save the optional literature-adaptive history.
        writetable(struct2table(raw_rows),fullfile(opt.output_dir,'raw_trial_results.csv'));
    end % Finish the low-rank trial loop.
end % Finish the condition loop.
writetable(struct2table(raw_rows),fullfile(opt.output_dir,'raw_trial_results.csv'));
summary = aggregate_results(raw_rows); % Aggregate every graph-rank-method group into mean and standard deviation.
writetable(summary,fullfile(opt.output_dir,'summary_mean_std.csv')); % Save the compact statistical summary.
conditions_saved = sanitize_conditions(conditions); % Remove machine-specific graph paths before saving raw results.
opt_saved = sanitize_options(opt); % Remove machine-specific option paths before saving raw results.
save(fullfile(opt.output_dir,'raw_results_and_assignments.mat'),'raw_rows','assignment_records','conditions_saved','opt_saved'); % Save raw values without machine-specific paths.
write_configuration(opt,conditions,graph_cache,fullfile(opt.output_dir,'config.json')); % Save the complete reproducibility configuration.
disp(summary); % Display the compact summary in the command window.
fprintf('Saved Max-Cut results to: %s\n',opt.output_dir); % Print the output location.
end % Finish the main benchmark function.

function validate_options(opt) % Validate benchmark options before any expensive computation.
if ~(isscalar(opt.num_trials) && opt.num_trials>=1 && opt.num_trials==floor(opt.num_trials)) % Validate the trial count.
    error('num_trials must be a positive integer.'); % Stop on an invalid trial count.
end % Finish the trial-count validation.
if ~(isscalar(opt.rounds) && opt.rounds>=1 && opt.rounds==floor(opt.rounds)) % Validate the rounding count.
    error('rounds must be a positive integer.'); % Stop on an invalid rounding count.
end % Finish the rounding-count validation.
if ~(isscalar(opt.max_iter) && opt.max_iter>=1 && opt.max_iter==floor(opt.max_iter)) % Validate the iteration limit.
    error('max_iter must be a positive integer.'); % Stop on an invalid iteration limit.
end % Finish the iteration-limit validation.
if ~(isscalar(opt.tol) && opt.tol>0) % Validate the stopping tolerance.
    error('tol must be positive.'); % Stop on an invalid stopping tolerance.
end % Finish the tolerance validation.
if ~(isscalar(opt.epsilon) && opt.epsilon>0) % Validate the gamma safeguard.
    error('epsilon must be positive.'); % Stop on an invalid gamma safeguard.
end % Finish the gamma-safeguard validation.
if ~(isscalar(opt.gamma_max) && opt.gamma_max>opt.epsilon) % Validate the upper gamma safeguard.
    error('gamma_max must exceed epsilon.'); % Stop on an invalid upper gamma safeguard.
end % Finish the upper-gamma validation.
if ~(isscalar(opt.rho0) && opt.rho0>0) % Validate the initial augmented-Lagrangian penalty.
    error('rho0 must be positive.'); % Stop on an invalid initial multiplier penalty.
end % Finish the initial-rho validation.
if ~(isscalar(opt.rho_growth) && opt.rho_growth>=1) % Validate the multiplier-penalty growth factor.
    error('rho_growth must be at least one.'); % Stop on an invalid multiplier-penalty growth factor.
end % Finish the rho-growth validation.
if ~(isscalar(opt.rho_max) && opt.rho_max>=opt.rho0) % Validate the multiplier-penalty upper safeguard.
    error('rho_max must be no smaller than rho0.'); % Stop on an invalid multiplier-penalty upper safeguard.
end % Finish the rho-maximum validation.
if ~(isscalar(opt.multiplier_update_period) && opt.multiplier_update_period>=1 && opt.multiplier_update_period==floor(opt.multiplier_update_period)) % Validate the multiplier update period.
    error('multiplier_update_period must be a positive integer.'); % Stop on an invalid multiplier update period.
end % Finish the multiplier-update-period validation.
if ~(isscalar(opt.constraint_tol) && opt.constraint_tol>0) % Validate the equality feasibility tolerance.
    error('constraint_tol must be positive.'); % Stop on an invalid equality feasibility tolerance.
end % Finish the equality-tolerance validation.
if ~(isscalar(opt.residual_reduction) && opt.residual_reduction>0 && opt.residual_reduction<1) % Validate the residual-reduction target.
    error('residual_reduction must lie strictly between zero and one.'); % Stop on an invalid residual-reduction target.
end % Finish the residual-reduction validation.
if ~(isscalar(opt.gamma_base) && opt.gamma_base>0) % Validate the normalized numerical gamma baseline.
    error('gamma_base must be positive.'); % Stop on an invalid gamma baseline.
end % Finish the gamma-baseline validation.
if ~(isscalar(opt.backtrack_factor) && opt.backtrack_factor>0 && opt.backtrack_factor<1) % Validate the backtracking factor.
    error('backtrack_factor must lie strictly between zero and one.'); % Stop on an invalid backtracking factor.
end % Finish the backtracking-factor validation.
if ~(isscalar(opt.progress_period) && opt.progress_period>=1 && opt.progress_period==floor(opt.progress_period)) % Validate the progress-message period.
    error('progress_period must be a positive integer.'); % Stop on an invalid progress-message period.
end % Finish the progress-period validation.
end % Finish the benchmark-option validation.

function conditions = build_conditions(opt) % Build an exact two-column cell array of graph sources and ranks.
if ~isempty(opt.condition_list) % Check whether an exact condition list was supplied.
    if ~iscell(opt.condition_list) || size(opt.condition_list,2)~=2 % Validate the condition-list format.
        error('condition_list must be an N-by-2 cell array containing graph source and rank.'); % Stop on an invalid condition list.
    end % Finish the exact-condition validation.
    conditions = opt.condition_list; % Use the exact graph-and-rank pairs.
    return; % Return after building exact conditions.
end % Finish the exact-condition branch.
graph_files = normalize_cellstr(opt.graph_files); % Normalize graph file inputs to a cell array of character vectors.
rank_values = opt.ranks(:)'; % Convert the rank list to a row vector.
if ~isempty(graph_files) % Check whether graph files were supplied.
    conditions = cell(numel(graph_files)*numel(rank_values),2); % Preallocate the Cartesian condition array.
    counter = 0; % Initialize the Cartesian condition counter.
    for graph_index = 1:numel(graph_files) % Loop over supplied graph files.
        for rank_index = 1:numel(rank_values) % Loop over requested ranks.
            counter = counter+1; % Advance the Cartesian condition counter.
            conditions{counter,1} = graph_files{graph_index}; % Store the graph-file source.
            conditions{counter,2} = rank_values(rank_index); % Store the low-rank dimension.
        end % Finish the rank loop.
    end % Finish the graph-file loop.
else % Handle the standalone synthetic fallback.
    sizes = opt.synthetic_sizes(:)'; % Convert synthetic sizes to a row vector.
    conditions = cell(numel(sizes)*numel(rank_values),2); % Preallocate synthetic graph-and-rank conditions.
    counter = 0; % Initialize the synthetic condition counter.
    for size_index = 1:numel(sizes) % Loop over synthetic graph sizes.
        for rank_index = 1:numel(rank_values) % Loop over requested ranks.
            counter = counter+1; % Advance the synthetic condition counter.
            conditions{counter,1} = sprintf('synthetic:%d:%d',sizes(size_index),opt.synthetic_seed+size_index-1); % Encode the synthetic graph source.
            conditions{counter,2} = rank_values(rank_index); % Store the low-rank dimension.
        end % Finish the rank loop.
    end % Finish the synthetic-size loop.
end % Finish the graph-source branch.
for condition_index = 1:size(conditions,1) % Loop over all built conditions.
    rank_value = conditions{condition_index,2}; % Read the current rank.
    if ~(isnumeric(rank_value) && isscalar(rank_value) && rank_value>=1 && rank_value==floor(rank_value)) % Validate the current rank.
        error('Every requested rank must be a positive integer.'); % Stop on an invalid rank.
    end % Finish the per-condition rank validation.
end % Finish the built-condition validation loop.
end % Finish the condition builder.

function values = normalize_cellstr(input_value) % Normalize text or cell text into a cell array of character vectors.
if isempty(input_value) % Check whether the input is empty.
    values = {}; % Return an empty cell array.
elseif ischar(input_value) % Check whether a single character vector was supplied.
    values = {input_value}; % Wrap the character vector in a cell array.
elseif iscell(input_value) % Check whether a cell array was supplied.
    values = input_value; % Copy the cell array.
else % Handle unsupported graph-file input types.
    error('graph_files must be a character vector or cell array of character vectors.'); % Stop on an invalid graph-file input.
end % Finish the graph-file normalization branch.
for index = 1:numel(values) % Loop over normalized graph-file entries.
    if ~ischar(values{index}) % Check whether the current entry is a character vector.
        error('Every graph_files entry must be a character vector.'); % Stop on a non-character graph source.
    end % Finish the per-entry graph-file validation.
end % Finish the normalized graph-file loop.
end % Finish the cell-string normalizer.

function key = graph_cache_key(source) % Build a stable cache key for a graph source.
if ischar(source) % Check whether the source is a character vector.
    key = source; % Use the source text as the cache key.
else % Handle an unsupported graph-source type.
    error('Graph sources must be character vectors.'); % Stop on an unsupported graph-source type.
end % Finish the graph-source type branch.
end % Finish the graph-cache key builder.

function graph = obtain_graph(source,opt,condition_index) % Load a graph file or generate a synthetic graph.
if strncmp(source,'synthetic:',10) % Check whether the source encodes a synthetic graph.
    tokens = sscanf(source,'synthetic:%f:%f'); % Parse the encoded synthetic size and seed as double-precision values.
    if numel(tokens)~=2 % Check whether the synthetic graph specification was parsed completely.
        error('Invalid synthetic graph specification: %s.',source); % Stop when the encoded size or seed is missing.
    end % Finish the synthetic graph specification validation.
    n = double(tokens(1)); % Read the synthetic graph size as a MATLAB double.
    seed = double(tokens(2)); % Read the synthetic graph seed as a MATLAB double.
    graph = generate_synthetic_graph(n,opt.synthetic_probability,seed); % Generate the synthetic positive-weight graph.
else % Handle a graph-file source.
    graph = load_graph_file(source); % Load a Gset-style or plain edge-list graph.
end % Finish the graph-source branch.
graph.source = source; % Record the original graph source.
graph.condition_index_first_seen = condition_index; % Record where the graph first appeared.
end % Finish the graph acquisition function.

function graph = generate_synthetic_graph(n,probability,seed) % Generate a sparse positive-weight undirected graph.
n = double(n); % Convert the graph size to double for compatibility with older MATLAB sparse-random routines.
probability = double(probability); % Convert the edge probability to double for compatibility with older MATLAB arithmetic.
seed = double(seed); % Convert the random seed to double for compatibility with older MATLAB random-stream routines.
if ~(isscalar(n) && isfinite(n) && n>=2 && n==floor(n)) % Validate the converted graph size.
    error('Synthetic graph size must be an integer greater than or equal to two.'); % Stop on an invalid synthetic graph size.
end % Finish the synthetic graph size validation.
if ~(isscalar(probability) && isfinite(probability) && probability>0 && probability<=1) % Validate the converted edge probability.
    error('Synthetic edge probability must lie in the interval (0,1].'); % Stop on an invalid synthetic edge probability.
end % Finish the synthetic edge-probability validation.
rng(seed,'twister'); % Set the synthetic graph random stream.
pattern = sprand(double(n),double(n),double(probability)); % Generate a sparse random edge pattern using double-precision dimensions.
pattern = triu(pattern,1); % Keep only strict upper-triangular candidate edges.
[row_indices,column_indices,~] = find(pattern); % Extract sampled edge endpoints.
weights = 0.5+rand(numel(row_indices),1); % Generate positive edge weights in the interval [0.5,1.5].
W = sparse([row_indices;column_indices],[column_indices;row_indices],[weights;weights],n,n); % Build the symmetric sparse adjacency matrix.
graph = graph_template(); % Start from the fixed graph structure template.
graph.name = sprintf('synthetic_n%d_p%.4f_seed%d',n,probability,seed); % Define the synthetic graph name.
graph.W = W; % Store the synthetic weighted adjacency matrix.
graph.n = n; % Store the number of vertices.
graph.m = nnz(triu(W,1)); % Store the number of undirected edges.
graph.weights_nonnegative = true; % Record that all generated weights are nonnegative.
end % Finish the synthetic graph generator.

function graph = load_graph_file(path) % Load a Gset-style graph or a plain weighted edge list.
file_identifier = fopen(path,'r'); % Open the graph file for reading.
if file_identifier<0 % Check whether the graph file opened successfully.
    error('Cannot open graph file: %s',path); % Stop when the graph file cannot be opened.
end % Finish the graph-file open validation.
cleanup_object = onCleanup(@()fclose(file_identifier)); %#ok<NASGU> % Guarantee that the graph file is closed.
numeric_rows = cell(0,1); % Initialize parsed numeric rows.
while true % Read the graph file line by line.
    line = fgetl(file_identifier); % Read the next graph-file line.
    if ~ischar(line) % Check whether the end of file was reached.
        break; % Exit the graph-file reading loop.
    end % Finish the end-of-file branch.
    line = strtrim(line); % Remove surrounding whitespace.
    if isempty(line) % Check whether the line is empty.
        continue; % Skip empty lines.
    end % Finish the empty-line branch.
    if line(1)=='#' || line(1)=='%' % Check whether the line is a comment.
        continue; % Skip comment lines.
    end % Finish the comment-line branch.
    values = sscanf(line,'%f')'; % Parse all numeric values from the current line.
    if numel(values)>=2 % Check whether the line contains at least two numbers.
        numeric_rows{end+1,1} = values; %#ok<AGROW> % Append the parsed numeric row.
    end % Finish the parsed-row branch.
end % Finish the graph-file reading loop.
if isempty(numeric_rows) % Check whether any graph data were parsed.
    error('No graph data were found in %s.',path); % Stop on an empty or unsupported graph file.
end % Finish the empty-graph validation.
has_header = false; % Initialize the Gset-header flag.
if numel(numeric_rows{1})==2 && numel(numeric_rows)>=2 % Check whether the first row could be an n-m header.
    candidate_n = round(numeric_rows{1}(1)); % Read the candidate vertex count.
    candidate_m = round(numeric_rows{1}(2)); % Read the candidate edge count.
    if candidate_n>=1 && candidate_m==numel(numeric_rows)-1 % Test the standard Gset header convention.
        has_header = true; % Mark the first row as a valid Gset header.
    end % Finish the Gset-header test.
end % Finish the possible-header branch.
if has_header % Handle the Gset-header format.
    n = candidate_n; % Use the declared vertex count.
    edge_rows = numeric_rows(2:end); % Remove the header row.
    one_based = true; % Treat Gset vertex identifiers as one-based.
else % Handle a plain edge-list format.
    edge_rows = numeric_rows; % Use every numeric row as an edge row.
    minimum_endpoint = inf; % Initialize the minimum endpoint.
    maximum_endpoint = -inf; % Initialize the maximum endpoint.
    for row_index = 1:numel(edge_rows) % Loop over edge rows to infer indexing.
        minimum_endpoint = min(minimum_endpoint,min(edge_rows{row_index}(1:2))); % Update the minimum endpoint.
        maximum_endpoint = max(maximum_endpoint,max(edge_rows{row_index}(1:2))); % Update the maximum endpoint.
    end % Finish the endpoint-inference loop.
    one_based = minimum_endpoint>=1; % Detect whether the edge list uses one-based indexing.
    n = round(maximum_endpoint)+(~one_based); % Infer the number of vertices.
end % Finish the graph-format branch.
row_indices = zeros(numel(edge_rows),1); % Preallocate first edge endpoints.
column_indices = zeros(numel(edge_rows),1); % Preallocate second edge endpoints.
weights = ones(numel(edge_rows),1); % Preallocate unit edge weights.
for edge_index = 1:numel(edge_rows) % Loop over parsed edges.
    values = edge_rows{edge_index}; % Read the current edge row.
    row_indices(edge_index) = round(values(1)); % Store the first endpoint.
    column_indices(edge_index) = round(values(2)); % Store the second endpoint.
    if numel(values)>=3 % Check whether an explicit edge weight exists.
        weights(edge_index) = values(3); % Store the explicit edge weight.
    end % Finish the edge-weight branch.
end % Finish the parsed-edge loop.
if ~one_based % Check whether zero-based identifiers must be converted.
    row_indices = row_indices+1; % Convert first endpoints to MATLAB indexing.
    column_indices = column_indices+1; % Convert second endpoints to MATLAB indexing.
end % Finish the indexing-conversion branch.
keep = row_indices~=column_indices; % Identify non-self-loop edges.
row_indices = row_indices(keep); % Remove self-loop first endpoints.
column_indices = column_indices(keep); % Remove self-loop second endpoints.
weights = weights(keep); % Remove self-loop weights.
W = sparse([row_indices;column_indices],[column_indices;row_indices],[weights;weights],n,n); % Build the symmetric sparse adjacency matrix.
W = 0.5*(W+W'); % Symmetrize duplicate or asymmetric input robustly.
W = W-spdiags(diag(W),0,n,n); % Remove all diagonal entries.
[~,base_name,~] = fileparts(path); % Extract the graph base name.
graph = graph_template(); % Start from the fixed graph structure template.
graph.name = base_name; % Store the graph base name for tables and best-known-cut lookup.
graph.W = W; % Store the weighted adjacency matrix.
graph.n = n; % Store the number of vertices.
graph.m = nnz(triu(W,1)); % Store the number of undirected edges.
upper_weights = nonzeros(triu(W,1)); % Extract unique undirected edge weights.
graph.weights_nonnegative = isempty(upper_weights) || min(upper_weights)>=0; % Record whether all edge weights are nonnegative.
end % Finish the graph-file loader.

function graph = prepare_graph(graph,opt) % Build original-scale and normalized Laplacian operators.
degree = full(sum(graph.W,2)); % Compute the weighted degree vector.
graph.L_original = spdiags(degree,0,graph.n,graph.n)-graph.W; % Construct the original weighted graph Laplacian.
if opt.require_nonnegative_weights && ~graph.weights_nonnegative % Check whether the caller explicitly requested nonnegative edge weights.
    error('Graph %s has negative edge weights while require_nonnegative_weights=true.',graph.name); % Stop only when the optional nonnegative-weight restriction was requested.
end % Finish the optional nonnegative-weight validation.
if opt.normalize_laplacian % Check whether optimization normalization is enabled.
    graph.scale = estimate_operator_norm(graph.L_original); % Estimate the Laplacian spectral norm.
else % Handle unnormalized optimization.
    graph.scale = 1; % Use the original Laplacian scale.
end % Finish the Laplacian-normalization branch.
if ~(isfinite(graph.scale) && graph.scale>0) % Check whether the estimated scale is usable.
    graph.scale = 1; % Fall back to a unit scale.
end % Finish the scale-safeguard branch.
graph.L_work = graph.L_original/graph.scale; % Build the normalized optimization Laplacian.
graph.W_work = graph.W/graph.scale; % Build the correspondingly normalized adjacency matrix.
graph.gradient_fro = norm(-0.25*graph.L_work,'fro'); % Record the initial lifted-gradient Frobenius norm.
graph.L_work_diagonal = full(diag(graph.L_work)); % Cache the normalized Laplacian diagonal for exact gradient-norm updates.
end % Finish the graph preparation function.

function value = estimate_operator_norm(A) % Compute a deterministic upper bound on the spectral norm.
row_absolute_sums = full(sum(abs(A),2)); % Compute the absolute row sums without forming a dense matrix.
if isempty(row_absolute_sums) % Check whether the matrix has no rows.
    value = 0; % Return zero for an empty matrix.
else % Handle a nonempty matrix.
    value = max(row_absolute_sums); % Use the induced infinity norm as a rigorous spectral-norm upper bound for symmetric A.
end % Finish the empty-matrix branch.
end % Finish the deterministic operator-norm bound.

function gamma = resolve_gamma_alm(value,gamma_base,gradient_fro,epsilon) % Resolve the common initial gamma for the decreasing gradient-norm experiment.
default_gamma = max([gamma_base,gradient_fro,epsilon]); % Start no lower than the initial lifted-gradient norm and the optional baseline.
if ischar(value) % Check whether gamma0 is a text specification.
    if strcmpi(value,'auto') % Check whether automatic initialization is requested.
        gamma = default_gamma; % Use the shared initial lifted-gradient scale.
    else % Handle numeric text.
        gamma = str2double(value); % Convert the numeric text to a scalar.
    end % Finish the text gamma branch.
else % Handle a numeric gamma input.
    gamma = value; % Copy the numeric initial gamma.
end % Finish the gamma-input branch.
if ~(isnumeric(gamma) && isscalar(gamma) && isfinite(gamma) && gamma>=epsilon) % Validate the resolved gamma.
    error('gamma0 must be ''auto'' or a finite numeric scalar no smaller than epsilon.'); % Stop on an invalid initial gamma.
end % Finish the resolved-gamma validation.
end % Finish the ALM gamma resolver.

function best_known = load_best_known(path) % Load optional best-known cut values from CSV.
best_known = containers.Map('KeyType','char','ValueType','double'); % Initialize an empty best-known-cut map.
if isempty(path) % Check whether no best-known file was supplied.
    return; % Return the empty map.
end % Finish the empty best-known-file branch.
if exist(path,'file')~=2 % Check whether the supplied file exists.
    error('Best-known CSV file not found: %s',path); % Stop on a missing best-known file.
end % Finish the best-known-file validation.
table_data = readtable(path); % Read the best-known cut table.
variable_names = table_data.Properties.VariableNames; % Read table variable names.
graph_column = find(strcmpi(variable_names,'graph_name'),1); % Locate the graph-name column.
cut_column = find(strcmpi(variable_names,'best_known_cut'),1); % Locate the best-known-cut column.
if isempty(graph_column) || isempty(cut_column) % Check whether required columns are present.
    error('The best-known CSV must contain graph_name and best_known_cut columns.'); % Stop on an unsupported best-known file.
end % Finish the required-column validation.
for row_index = 1:height(table_data) % Loop over best-known entries.
    graph_value = table_data{row_index,graph_column}; % Read the graph-name value.
    if iscell(graph_value) % Check whether the graph name is returned inside a cell.
        graph_name = graph_value{1}; % Extract the graph-name character vector.
    else % Handle direct character or string-like table content.
        graph_name = char(graph_value); % Convert the graph name to a character vector.
    end % Finish the graph-name conversion branch.
    cut_value = table_data{row_index,cut_column}; % Read the best-known cut value.
    if iscell(cut_value) % Check whether the cut value is returned inside a cell.
        cut_value = cut_value{1}; % Extract the numeric cut value.
    end % Finish the cut-value conversion branch.
    if isnumeric(cut_value) && isscalar(cut_value) && isfinite(cut_value) % Validate the best-known cut value.
        best_known(graph_name) = double(cut_value); % Store the valid best-known cut value.
    end % Finish the valid best-known entry branch.
end % Finish the best-known entry loop.
end % Finish the best-known cut loader.

function value = lookup_best_known(best_known,graph_name) % Retrieve a best-known cut value for one graph.
if isKey(best_known,graph_name) % Check whether the graph has a recorded best-known cut.
    value = best_known(graph_name); % Return the recorded best-known cut.
else % Handle a graph without a supplied best-known value.
    value = NaN; % Return a missing cut ratio reference.
end % Finish the best-known lookup branch.
end % Finish the best-known cut lookup.

function [result,history,best_assignment] = solve_sbmf_pgd(graph,X0,opt,hyperplanes,best_known) % Solve the low-rank Max-Cut SDP by Riemannian projected gradient.
X = X0; % Copy the common feasible initialization.
current_value = factor_sdp_value(graph.L_work,X); % Evaluate the initial normalized relaxation value.
records = nan(opt.max_iter,5); % Preallocate iteration, time, objective, gradient, and relative-change history.
start_time = tic; % Start the wall-clock timer.
last_iteration = opt.max_iter; % Initialize the reported iteration count.
step = opt.sbmf_step; % Initialize the projected-gradient step.
for iteration = 1:opt.max_iter % Start the projected-gradient iteration loop.
    euclidean_gradient = 0.5*(graph.L_work*X); % Compute the Euclidean ascent gradient.
    row_inner = sum(euclidean_gradient.*X,2); % Compute rowwise radial gradient components.
    tangent_gradient = euclidean_gradient-row_inner.*X; % Project the gradient onto the product-sphere tangent space.
    gradient_norm = norm(tangent_gradient,'fro'); % Compute the Riemannian gradient norm.
    trial_step = step; % Initialize the current backtracking trial step.
    candidate = X; % Initialize the candidate factor.
    candidate_value = current_value; % Initialize the candidate objective.
    for backtrack_index = 1:30 % Limit the number of backtracking reductions.
        candidate = normalize_rows(X+trial_step*tangent_gradient); % Retract the tangent step onto unit row spheres.
        candidate_value = factor_sdp_value(graph.L_work,candidate); % Evaluate the candidate normalized relaxation value.
        sufficient_increase = current_value+opt.armijo_constant*trial_step*gradient_norm^2; % Form the Armijo sufficient-increase threshold.
        if candidate_value>=sufficient_increase || trial_step<=1e-14 % Check the ascent acceptance condition.
            break; % Accept the current candidate.
        end % Finish the ascent acceptance branch.
        trial_step = trial_step*opt.backtrack_factor; % Reduce the backtracking step.
    end % Finish the backtracking loop.
    X = candidate; % Accept the projected-gradient candidate.
    relative_change = abs(candidate_value-current_value)/max(1,abs(current_value)); % Compute the relative relaxation-value change.
    current_value = candidate_value; % Update the current normalized relaxation value.
    step = min(trial_step/opt.backtrack_factor,opt.sbmf_step); % Modestly restore the next initial step after successful backtracking.
    elapsed = toc(start_time); % Measure elapsed wall-clock time.
    records(iteration,:) = [iteration,elapsed,current_value,gradient_norm,relative_change]; % Record the current projected-gradient state.
    print_iteration_progress(opt,'sBMF-PGD',iteration,opt.max_iter,elapsed,current_value,relative_change,gradient_norm,NaN,NaN); % Print periodic projected-gradient progress.
    if relative_change<opt.tol && gradient_norm<sqrt(opt.tol) % Check the projected-gradient stopping criterion.
        last_iteration = iteration; % Record the stopping iteration.
        break; % Exit the projected-gradient loop.
    end % Finish the projected-gradient stopping branch.
end % Finish the projected-gradient loop.
elapsed = toc(start_time); % Measure the final projected-gradient runtime.
feasible_factor = normalize_rows(X); % Form the common feasible factor used by every evaluation metric.
[best_cut,mean_cut,best_assignment] = round_factor(graph.L_original,feasible_factor,hyperplanes); % Apply shared random-hyperplane rounding.
result = build_result(graph,size(X,2),'sBMF-PGD',last_iteration,elapsed,NaN,feasible_factor,feasible_factor,best_cut,mean_cut,best_known); % Build the projected-gradient result row.
history = array2table(records(1:last_iteration,:),'VariableNames',{'iteration','time','sdp_objective_work','gradient_norm','relative_change'}); % Convert the projected-gradient history to a table.
end % Finish the symmetric projected-gradient solver.

function [result,history,best_assignment] = solve_sbmf_mixing(graph,X0,opt,hyperplanes,best_known) % Solve the low-rank Max-Cut SDP by cyclic Mixing updates.
X = X0; % Copy the common feasible initialization.
neighbor_sum = graph.W_work*X; % Maintain weighted neighbor sums for efficient coordinate updates.
current_value = factor_sdp_value(graph.L_work,X); % Evaluate the initial normalized relaxation value.
records = nan(opt.max_iter,4); % Preallocate sweep, time, objective, and relative-change history.
start_time = tic; % Start the wall-clock timer.
last_iteration = opt.max_iter; % Initialize the reported sweep count.
for sweep = 1:opt.max_iter % Start the cyclic Mixing sweep loop.
    for vertex = 1:graph.n % Update every vertex factor row cyclically.
        direction = -neighbor_sum(vertex,:); % Form the coordinate-ascent direction from off-diagonal Laplacian terms.
        direction_norm = norm(direction); % Compute the coordinate direction norm.
        if direction_norm>1e-14 % Check whether a meaningful coordinate direction exists.
            new_row = direction/direction_norm; % Maximize the coordinate linear objective over the unit sphere.
            delta = new_row-X(vertex,:); % Compute the accepted row change.
            if any(delta~=0) % Check whether the row changed.
                X(vertex,:) = new_row; % Store the new row.
                [neighbor_indices,~,neighbor_weights] = find(graph.W_work(:,vertex)); % Read only the nonzero neighbors of the changed vertex.
                if ~isempty(neighbor_indices) % Check whether the changed vertex has weighted neighbors.
                    neighbor_sum(neighbor_indices,:) = neighbor_sum(neighbor_indices,:)+neighbor_weights*delta; % Update only affected neighbor sums.
                end % Finish the sparse-neighbor update branch.
            end % Finish the changed-row branch.
        end % Finish the meaningful-direction branch.
    end % Finish the vertex-update loop.
    candidate_value = factor_sdp_value(graph.L_work,X); % Evaluate the relaxation value after the current sweep.
    relative_change = abs(candidate_value-current_value)/max(1,abs(current_value)); % Compute the relative sweep improvement.
    current_value = candidate_value; % Update the current relaxation value.
    elapsed = toc(start_time); % Measure elapsed wall-clock time.
    records(sweep,:) = [sweep,elapsed,current_value,relative_change]; % Record the current Mixing sweep.
    print_iteration_progress(opt,'sBMF-Mixing',sweep,opt.max_iter,elapsed,current_value,relative_change,NaN,NaN,NaN); % Print periodic Mixing progress.
    if relative_change<opt.tol % Check the Mixing stopping criterion.
        last_iteration = sweep; % Record the stopping sweep.
        break; % Exit the Mixing sweep loop.
    end % Finish the Mixing stopping branch.
end % Finish the Mixing sweep loop.
elapsed = toc(start_time); % Measure the final Mixing runtime.
feasible_factor = normalize_rows(X); % Form the common feasible factor used by every evaluation metric.
[best_cut,mean_cut,best_assignment] = round_factor(graph.L_original,feasible_factor,hyperplanes); % Apply shared random-hyperplane rounding.
result = build_result(graph,size(X,2),'sBMF-Mixing',last_iteration,elapsed,NaN,feasible_factor,feasible_factor,best_cut,mean_cut,best_known); % Build the Mixing result row.
history = array2table(records(1:last_iteration,:),'VariableNames',{'iteration','time','sdp_objective_work','relative_change'}); % Convert the Mixing history to a table.
end % Finish the symmetric Mixing solver.

function [result,history,best_assignment] = solve_abmf_alm(graph,X0,opt,hyperplanes,best_known,mode,gamma_initial) % Solve the standard Max-Cut SDP through a constrained asymmetric augmented Lagrangian.
if strcmpi(mode,'our')
    [result,history,best_assignment]=solve_abmf_alm_mix(graph,X0,opt,hyperplanes,best_known,gamma_initial);return;
end
X = X0; % Initialize the first factor from the shared unit-row factor.
Y = X0; % Initialize the second factor identically.
lambda = zeros(graph.n,1); % Initialize the diagonal-equality multiplier.
rho = opt.rho0; % Initialize the augmented-Lagrangian feasibility penalty.
gamma = max(gamma_initial,opt.epsilon); % Safeguard the initial agreement penalty.
gamma_cap_hit = false; % Initialize the agreement-penalty cap diagnostic.
rho_cap_hit = false; % Initialize the feasibility-penalty cap diagnostic.
previous_multiplier_residual = Inf; % Initialize the residual used by the rho update.
common_factor = normalize_rows(0.5*(X+Y)); % Form the initial feasible symmetric evaluation factor.
current_value = factor_sdp_value(graph.L_work,common_factor); % Evaluate the initial normalized feasible SDP value.
records = nan(opt.max_iter,14); % Preallocate the constrained asymmetric history.
start_time = tic; % Start the wall-clock timer.
last_iteration = opt.max_iter; % Initialize the reported iteration count.
gamma_candidate = gamma; % Initialize the decreasing gradient-norm candidate.
gradient_fro = graph.gradient_fro; % Initialize the Frobenius norm of the lifted gradient.
for iteration = 1:opt.max_iter % Start the alternating augmented-Lagrangian sweep loop.
    X = solve_alm_block(Y,graph.L_work*Y,lambda,rho,gamma); % Exactly solve the strongly convex X block.
    Y = solve_alm_block(X,graph.L_work*X,lambda,rho,gamma); % Exactly solve the strongly convex Y block.
    constraint_vector = sum(X.*Y,2)-1; % Compute diag(X*Y')-1 without forming the dense product.
    equality_residual = norm(constraint_vector,Inf); % Measure the standard SDP diagonal-equality residual.
    augmented_gradient_diagonal = lambda+rho*constraint_vector; % Form the diagonal term in grad_Z of the current augmented objective.
    gradient_fro = augmented_gradient_fro(graph,augmented_gradient_diagonal); % Compute the full lifted-gradient Frobenius norm without forming a dense matrix.
    gamma_candidate = max(gradient_fro,opt.epsilon); % Safeguard the current decreasing gamma candidate.
    if strcmpi(mode,'adp') % Check whether the literature adaptive rule is requested.
        gamma_next = update_gamma_adp(gamma,X,Y,opt.epsilon,opt.gamma_max); % Apply the multiplicative comparison rule.
        method_name = 'aBMF-adp'; % Set the comparison method name.
    else % Handle an unsupported asymmetric mode.
        error('Unsupported asymmetric gamma mode: %s',mode); % Stop on an unknown asymmetric mode.
    end % Finish the agreement-penalty branch.
    if gamma_next>opt.gamma_max % Check whether the numerical gamma cap would be exceeded.
        gamma_next = opt.gamma_max; % Apply the numerical gamma cap.
        gamma_cap_hit = true; % Record that the gamma cap was active.
    end % Finish the gamma-cap branch.
    gamma = gamma_next; % Accept the next agreement penalty.
    gamma_candidate = max(gamma_candidate,opt.epsilon); % Record the safeguarded current gradient-norm candidate.
    if mod(iteration,opt.multiplier_update_period)==0 % Check whether the equality multiplier should be updated.
        lambda = lambda+rho*constraint_vector; % Apply the standard augmented-Lagrangian multiplier update.
        if equality_residual>opt.constraint_tol && equality_residual>opt.residual_reduction*previous_multiplier_residual % Check whether an unresolved residual decreased sufficiently.
            rho_next = min(opt.rho_max,opt.rho_growth*rho); % Increase the feasibility penalty when progress is insufficient.
            if rho_next>=opt.rho_max && rho<opt.rho_max % Check whether the rho cap becomes active.
                rho_cap_hit = true; % Record that the rho cap was reached.
            end % Finish the rho-cap diagnostic branch.
            rho = rho_next; % Accept the updated feasibility penalty.
        end % Finish the rho-growth branch.
        previous_multiplier_residual = equality_residual; % Store the current residual for the next multiplier update.
    end % Finish the multiplier-update branch.
    rel_disagreement = relative_disagreement(X,Y); % Compute the current relative factor disagreement.
    common_factor = normalize_rows(0.5*(X+Y)); % Form the standard feasible PSD factor used for cross-method evaluation.
    candidate_value = factor_sdp_value(graph.L_work,common_factor); % Evaluate the feasible normalized relaxation value.
    bilinear_value = bilinear_sdp_value(graph.L_work,X,Y); % Evaluate the asymmetric bilinear relaxation value.
    augmented_objective = -bilinear_value+sum(lambda.*constraint_vector)+0.5*rho*sum(constraint_vector.^2)+0.5*gamma*norm(X-Y,'fro')^2; % Evaluate the current augmented minimization objective.
    relative_change = abs(candidate_value-current_value)/max(1,abs(current_value)); % Compute the relative feasible-objective change.
    current_value = candidate_value; % Update the current feasible relaxation value.
    elapsed = toc(start_time); % Measure elapsed wall-clock time.
    records(iteration,:) = [iteration,elapsed,current_value,bilinear_value,augmented_objective,rel_disagreement,equality_residual,gamma,gamma_candidate,gradient_fro,rho,norm(lambda,Inf),relative_change,double(mod(iteration,opt.multiplier_update_period)==0)]; % Record the current constrained asymmetric state.
    print_iteration_progress(opt,method_name,iteration,opt.max_iter,elapsed,current_value,relative_change,rel_disagreement,equality_residual,gamma); % Print periodic constrained asymmetric progress.
    if relative_change<opt.tol && rel_disagreement<max(1e-4,sqrt(opt.tol)) && equality_residual<opt.constraint_tol % Check the constrained asymmetric stopping criterion.
        last_iteration = iteration; % Record the stopping iteration.
        break; % Exit the alternating augmented-Lagrangian loop.
    end % Finish the constrained stopping branch.
end % Finish the alternating augmented-Lagrangian loop.
elapsed = toc(start_time); % Measure the final constrained asymmetric runtime.
feasible_factor = normalize_rows(0.5*(X+Y)); % Form the standard feasible PSD factor for final evaluation.
[best_cut,mean_cut,best_assignment] = round_factor(graph.L_original,feasible_factor,hyperplanes); % Apply shared random-hyperplane rounding.
result = build_result(graph,size(X,2),method_name,last_iteration,elapsed,gamma,X,Y,best_cut,mean_cut,best_known); % Build the constrained asymmetric result row.
result.sdp_objective = factor_sdp_value(graph.L_original,feasible_factor); % Store the feasible symmetric SDP value.
result.bilinear_sdp_objective = bilinear_sdp_value(graph.L_original,X,Y); % Store the asymmetric bilinear value before feasible-factor conversion.
result.equality_residual = norm(sum(X.*Y,2)-1,Inf); % Store the standard SDP diagonal-equality residual.
result.multiplier_norm_inf = norm(lambda,Inf); % Store the final equality-multiplier norm.
result.final_rho = rho; % Store the final augmented-Lagrangian feasibility penalty.
result.gamma_candidate = gamma_candidate; % Store the final decreasing gradient-norm candidate.
result.gradient_fro = gradient_fro; % Store the final full lifted-gradient Frobenius norm.
result.gamma_cap_hit = double(gamma_cap_hit); % Store whether the gamma cap was active.
result.rho_cap_hit = double(rho_cap_hit); % Store whether the rho cap was active.
history = array2table(records(1:last_iteration,:),'VariableNames',{'iteration','time','feasible_sdp_objective_work','bilinear_sdp_objective_work','augmented_objective_work','rel_disagreement','equality_residual','gamma','gamma_candidate','gradient_fro','rho','multiplier_norm_inf','relative_change','multiplier_updated'}); % Convert the constrained asymmetric history to a table.
end % Finish the constrained asymmetric ALM solver.

function [result,history,best_assignment] = solve_abmf_alm_mix(graph,X0,opt,hyperplanes,best_known,gamma_initial)
X=X0;Y=X0;lambda=zeros(graph.n,1);rho=opt.rho0;gamma=max(gamma_initial,opt.epsilon);
rho_cap_hit=false;previous_multiplier_residual=Inf;
current_value=factor_sdp_value(graph.L_work,normalize_rows(0.5*(X+Y)));
LY=graph.L_work*Y;
names={'iteration','time','feasible_sdp_objective_work','bilinear_sdp_objective_work', ...
 'augmented_objective_work','rel_disagreement','equality_residual','gamma','gamma_candidate', ...
 'gradient_fro','rho','multiplier_norm_inf','relative_change','multiplier_updated', ...
 'gamma_used','gamma_previous','gamma_next','gamma_hat','gamma_check','directional_quotient', ...
 'loss_gradient_fro','D_zero','D_norm_sq'};
records=nan(opt.max_iter,numel(names));clock_start=tic;stop_reason='iteration_limit';
for iteration=1:opt.max_iter
    used=gamma;
    X=solve_alm_block(Y,LY,lambda,rho,used);
    LX=graph.L_work*X;
    Y=solve_alm_block(X,LX,lambda,rho,used);
    LY=graph.L_work*Y;
    constraint_vector=sum(X.*Y,2)-1;equality_residual=norm(constraint_vector,Inf);
    if mod(iteration,opt.multiplier_update_period)==0
        lambda=lambda+rho*constraint_vector;
        if equality_residual>opt.constraint_tol&&equality_residual>opt.residual_reduction*previous_multiplier_residual
            rho_next=min(opt.rho_max,opt.rho_growth*rho);
            if rho_next>=opt.rho_max&&rho<opt.rho_max,rho_cap_hit=true;end
            rho=rho_next;
        end
        previous_multiplier_residual=equality_residual;
    end
    % Freeze the updated lambda/rho while evaluating the next ALM-loss gradient.
    diagonal=lambda+rho*constraint_vector;
    [next,info]=gamma_maxcut_mix(graph,X,Y,diagonal,used,opt.gamma_nu,opt.gamma_mix_epsilon0,LX,LY);
    rel_disagreement=relative_disagreement(X,Y);
    common_factor=normalize_rows(0.5*(X+Y));candidate_value=factor_sdp_value(graph.L_work,common_factor);
    bilinear_value=bilinear_sdp_value(graph.L_work,X,Y);
    augmented=-bilinear_value+sum(lambda.*constraint_vector)+0.5*rho*sum(constraint_vector.^2)+0.5*used*norm(X-Y,'fro')^2;
    relative_change=abs(candidate_value-current_value)/max(1,abs(current_value));current_value=candidate_value;
    records(iteration,:)=[iteration,toc(clock_start),current_value,bilinear_value,augmented,rel_disagreement,equality_residual, ...
        used,info.gamma_hat,info.loss_gradient_fro,rho,norm(lambda,Inf),relative_change,double(mod(iteration,opt.multiplier_update_period)==0), ...
        used,used,next,info.gamma_hat,info.gamma_check,info.directional_quotient,info.loss_gradient_fro,info.D_zero,info.D_norm_sq];
    print_iteration_progress(opt,'aBMF-our',iteration,opt.max_iter,toc(clock_start),current_value,relative_change,rel_disagreement,equality_residual,used);
    gamma=next;
    if relative_change<opt.tol&&rel_disagreement<max(1e-4,sqrt(opt.tol))&&equality_residual<opt.constraint_tol
        stop_reason='original_joint_tolerance';break;
    end
end
elapsed=toc(clock_start);feasible_factor=normalize_rows(0.5*(X+Y));
[best_cut,mean_cut,best_assignment]=round_factor(graph.L_original,feasible_factor,hyperplanes);
result=build_result(graph,size(X,2),'aBMF-our',iteration,elapsed,used,X,Y,best_cut,mean_cut,best_known);
result.sdp_objective=factor_sdp_value(graph.L_original,feasible_factor);
result.bilinear_sdp_objective=bilinear_sdp_value(graph.L_original,X,Y);
result.equality_residual=equality_residual;result.multiplier_norm_inf=norm(lambda,Inf);result.final_rho=rho;
result.gamma_candidate=info.gamma_hat;result.gradient_fro=info.loss_gradient_fro;
result.gamma_cap_hit=0;result.rho_cap_hit=double(rho_cap_hit);
result.final_gamma_next=next;result.stop_reason=stop_reason;
history=array2table(records(1:iteration,:),'VariableNames',names);
audit_mixed_history(history,opt.gamma_nu,opt.gamma_mix_epsilon0);
end

function X = solve_alm_block(anchor,Lanchor,lambda,rho,gamma) % Exactly solve one strongly convex asymmetric augmented-Lagrangian block.
row_norm_squared = sum(anchor.*anchor,2); % Compute the squared anchor-row norms.
right_hand_side = bsxfun(@times,gamma+rho-lambda,anchor)+0.25*Lanchor; % Form the rowwise linear-system right-hand side.
anchor_dot_rhs = sum(anchor.*right_hand_side,2); % Compute the rank-one Sherman-Morrison numerator.
denominator = gamma+rho*row_norm_squared; % Form the positive Sherman-Morrison denominator.
rank_one_term = bsxfun(@times,rho*anchor,anchor_dot_rhs./denominator); % Form the rowwise rank-one correction.
X = (right_hand_side-rank_one_term)/gamma; % Apply the closed-form inverse of gamma*I+rho*anchor*anchor'.
end % Finish the exact augmented-Lagrangian block solver.

function gradient_fro = augmented_gradient_fro(graph,gradient_diagonal) % Compute ||-L/4+Diag(gradient_diagonal)||_F exactly and efficiently.
linear_norm_squared = graph.gradient_fro^2; % Read the squared Frobenius norm of the linear loss gradient -L/4.
diagonal_norm_squared = sum(gradient_diagonal.^2); % Compute the squared Frobenius norm of the diagonal ALM term.
cross_term = -0.5*sum(graph.L_work_diagonal.*gradient_diagonal); % Compute twice the Frobenius inner product between -L/4 and the diagonal term.
gradient_norm_squared = linear_norm_squared+diagonal_norm_squared+cross_term; % Combine the exact squared-norm terms.
gradient_fro = sqrt(max(0,gradient_norm_squared)); % Return a real nonnegative Frobenius norm despite roundoff.
end % Finish the augmented lifted-gradient norm helper.

function gamma = update_gamma_adp(previous,X,Y,epsilon,gamma_max) % Apply the comparison multiplicative adaptive gamma update.
numerator = sum(X(:).^2)+sum(Y(:).^2); % Compute the sum of squared factor norms.
denominator = 2*abs(sum(X(:).*Y(:))); % Compute twice the absolute factor inner product.
ratio = numerator/max(denominator,1e-30); % Form the safeguarded multiplicative ratio.
gamma = min(max(previous*max(1,ratio),epsilon),gamma_max); % Update monotonically and clamp the comparison gamma.
end % Finish the literature adaptive gamma update.

function print_method_start(opt,method_name,trial,num_trials) % Print a method-start progress message.
if ~opt.verbose % Check whether progress output is disabled.
    return; % Return without printing when progress output is disabled.
end % Finish the progress-output branch.
if trial>0 % Check whether this is a low-rank trial method.
    fprintf('  -> %s, trial %d/%d started.\n',method_name,trial,num_trials); % Print the low-rank method start.
else % Handle the one-time full-SDP reference.
    fprintf('  -> %s started.\n',method_name); % Print the full-SDP method start.
end % Finish the trial-label branch.
drawnow; % Flush the method-start message.
end % Finish the method-start printer.

function print_method_done(opt,result) % Print a method-completion progress message.
if ~opt.verbose % Check whether progress output is disabled.
    return; % Return without printing when progress output is disabled.
end % Finish the progress-output branch.
if isfinite(result.final_gamma) % Check whether the result has an asymmetric agreement penalty.
    fprintf('  <- %s finished: iter=%d, time=%.2f s, SDP=%.6g, cut=%.6g, eq-res=%.3e, gamma=%.3e.\n',result.method,result.iterations,result.time,result.sdp_objective,result.best_cut,result.equality_residual,result.final_gamma); % Print the asymmetric completion summary.
else % Handle symmetric methods and the full-SDP reference.
    fprintf('  <- %s finished: iter=%d, time=%.2f s, SDP=%.6g, cut=%.6g.\n',result.method,result.iterations,result.time,result.sdp_objective,result.best_cut); % Print the symmetric completion summary.
end % Finish the asymmetric-result branch.
drawnow; % Flush the method-completion message.
end % Finish the method-completion printer.

function print_iteration_progress(opt,method_name,iteration,max_iterations,elapsed,objective,relative_change,diagnostic_one,diagnostic_two,gamma) % Print periodic within-method progress.
if ~opt.verbose % Check whether progress output is disabled.
    return; % Return without printing when progress output is disabled.
end % Finish the progress-output branch.
should_print = iteration==1 || mod(iteration,opt.progress_period)==0 || iteration==max_iterations; % Decide whether this iteration should be displayed.
if ~should_print % Check whether the current iteration is between reporting points.
    return; % Skip the current progress message.
end % Finish the reporting-period branch.
if strncmp(method_name,'aBMF',4) % Check whether constrained asymmetric diagnostics are available.
    fprintf('     %s iter %d/%d: time=%.1f s, SDP=%.6g, relchg=%.3e, X-Y=%.3e, eq-res=%.3e, gamma=%.3e.\n',method_name,iteration,max_iterations,elapsed,objective,relative_change,diagnostic_one,diagnostic_two,gamma); % Print asymmetric iteration progress.
elseif strcmp(method_name,'sBMF-PGD') % Check whether the projected-gradient norm is available.
    fprintf('     %s iter %d/%d: time=%.1f s, SDP=%.6g, relchg=%.3e, grad=%.3e.\n',method_name,iteration,max_iterations,elapsed,objective,relative_change,diagnostic_one); % Print projected-gradient iteration progress.
else % Handle Mixing and full-SDP progress.
    fprintf('     %s iter %d/%d: time=%.1f s, SDP=%.6g, relchg=%.3e.\n',method_name,iteration,max_iterations,elapsed,objective,relative_change); % Print generic iteration progress.
end % Finish the method-specific progress branch.
drawnow; % Flush the iteration progress message.
end % Finish the within-method progress printer.

function X = normalize_rows(M) % Normalize every row to unit Euclidean norm.
row_norms = sqrt(sum(M.*M,2)); % Compute all row Euclidean norms.
zero_rows = row_norms<=1e-15; % Identify numerically zero rows.
if any(zero_rows) % Check whether any rows require deterministic repair.
    M(zero_rows,:) = 0; % Clear all numerically zero rows.
    M(zero_rows,1) = 1; % Assign the first coordinate to create a deterministic unit row.
    row_norms(zero_rows) = 1; % Set repaired row norms to one.
end % Finish the zero-row repair branch.
X = M./row_norms; % Normalize every row.
end % Finish the row normalizer.

function value = factor_sdp_value(L,U) % Evaluate the standard feasible Max-Cut SDP objective from a factor.
value = 0.25*sum(sum(U.*(L*U))); % Compute 0.25*trace(U'*L*U).
end % Finish the factorized SDP objective.

function value = bilinear_sdp_value(L,X,Y) % Evaluate the asymmetric bilinear Max-Cut objective.
value = 0.25*sum(sum(X.*(L*Y))); % Compute 0.25*trace(X'*L*Y).
end % Finish the asymmetric bilinear objective.

function value = matrix_sdp_value(L,Z) % Evaluate the standard Max-Cut SDP objective from a full matrix.
value = 0.25*sum(sum(full(L).*Z)); % Compute 0.25*<L,Z>.
end % Finish the full-matrix SDP objective.

function value = relative_disagreement(X,Y) % Compute the scale-normalized factor disagreement.
scale = max([1,norm(X,'fro'),norm(Y,'fro')]); % Build a stable factor scale.
value = norm(X-Y,'fro')/scale; % Compute the relative disagreement.
end % Finish the relative-disagreement metric.

function [best_cut,mean_cut,best_assignment] = round_factor(L,U,hyperplanes) % Apply shared Goemans-Williamson random-hyperplane rounding.
sign_matrix = sign(U*hyperplanes); % Compute all rounded sign assignments.
sign_matrix(sign_matrix==0) = 1; % Resolve zero projections deterministically.
cut_values = 0.25*sum(sign_matrix.*(L*sign_matrix),1); % Evaluate every rounded cut using the original Laplacian.
[best_cut,best_index] = max(cut_values); % Identify the best rounded cut.
mean_cut = mean(cut_values); % Compute the mean rounded cut.
best_assignment = sign_matrix(:,best_index); % Return the best rounded sign assignment.
end % Finish the random-hyperplane rounding function.

function result = build_result(graph,rank_value,method,iterations,time_value,final_gamma,X,Y,best_cut,mean_cut,best_known) % Build one fixed-layout raw result row.
result = result_row_template(); % Start from the fixed raw result template.
result.graph_name = graph.name; % Store the graph name.
result.n = graph.n; % Store the number of vertices.
result.m = graph.m; % Store the number of edges.
result.rank = rank_value; % Store the low-rank dimension.
result.method = method; % Store the method name.
result.iterations = iterations; % Store the iteration or sweep count.
result.time = time_value; % Store wall-clock runtime.
result.final_gamma = final_gamma; % Store the final gamma when applicable.
common_factor = normalize_rows(0.5*(X+Y)); % Form the unified feasible factor for all cross-method metrics.
result.sdp_objective = factor_sdp_value(graph.L_original,common_factor); % Store the original-scale feasible relaxation value.
result.best_cut = best_cut; % Store the best rounded cut.
result.mean_cut = mean_cut; % Store the mean rounded cut.
reference_cut = lookup_best_known(best_known,graph.name); % Retrieve the optional best-known cut.
if isfinite(reference_cut) && reference_cut~=0 % Check whether a valid best-known denominator exists.
    result.cut_ratio = best_cut/reference_cut; % Store the ratio to the supplied best-known cut.
else % Handle a missing best-known cut.
    result.cut_ratio = NaN; % Store a missing cut ratio.
end % Finish the cut-ratio branch.
result.rel_disagreement = relative_disagreement(X,Y); % Store the relative factor disagreement.
result.bilinear_sdp_objective = result.sdp_objective; % Use the same value for symmetric methods.
result.equality_residual = 0; % Record exact diagonal feasibility for symmetric factors.
result.multiplier_norm_inf = NaN; % Mark the equality-multiplier metric as inapplicable.
result.final_rho = NaN; % Mark the augmented-Lagrangian penalty as inapplicable.
result.gamma_candidate = NaN; % Mark the gradient-norm gamma candidate as inapplicable.
result.gradient_fro = NaN; % Mark the augmented-gradient diagnostic as inapplicable.
result.gamma_cap_hit = 0; % Record that no asymmetric gamma cap applies.
result.rho_cap_hit = 0; % Record that no asymmetric rho cap applies.
end % Finish the raw result-row builder.

function [result,history,best_assignment] = solve_full_sdp_pg(graph,opt,rounding_seed,best_known) % Solve the dense full Max-Cut SDP by projected gradient and Dykstra projection.
n = graph.n; % Read the graph dimension.
Z = eye(n); % Initialize at the feasible identity correlation matrix.
current_value = matrix_sdp_value(graph.L_work,Z); % Evaluate the initial normalized SDP objective.
records = nan(opt.sdp_max_iter,4); % Preallocate iteration, time, objective, and relative-change history.
start_time = tic; % Start the full-SDP wall-clock timer.
last_iteration = opt.sdp_max_iter; % Initialize the reported full-SDP iteration count.
dense_gradient = 0.25*full(graph.L_work); % Form the constant dense ascent gradient.
for iteration = 1:opt.sdp_max_iter % Start the full-SDP projected-gradient loop.
    trial = Z+opt.sdp_step*dense_gradient; % Take a dense gradient-ascent step.
    Z = project_elliptope_dykstra(trial,opt.dykstra_max_iter,opt.dykstra_tol); % Project onto PSD matrices with unit diagonal.
    candidate_value = matrix_sdp_value(graph.L_work,Z); % Evaluate the projected normalized SDP objective.
    relative_change = abs(candidate_value-current_value)/max(1,abs(current_value)); % Compute the relative SDP objective change.
    current_value = candidate_value; % Update the normalized SDP objective.
    elapsed = toc(start_time); % Measure elapsed full-SDP time.
    records(iteration,:) = [iteration,elapsed,current_value,relative_change]; % Record the current full-SDP state.
    if relative_change<opt.tol % Check the full-SDP stopping criterion.
        last_iteration = iteration; % Record the stopping iteration.
        break; % Exit the full-SDP projected-gradient loop.
    end % Finish the full-SDP stopping branch.
end % Finish the full-SDP projected-gradient loop.
elapsed = toc(start_time); % Measure the final full-SDP runtime.
[U,eigenvalues] = factorize_psd_matrix(Z); % Factorize the final feasible correlation matrix.
rng(rounding_seed,'twister'); % Set the deterministic full-SDP rounding stream.
hyperplanes = randn(size(U,2),opt.rounds); % Generate random hyperplanes in the full-SDP factor dimension.
[best_cut,mean_cut,best_assignment] = round_factor(graph.L_original,U,hyperplanes); % Round the full-SDP solution.
result = result_row_template(); % Start from the fixed raw result template.
result.graph_name = graph.name; % Store the graph name.
result.n = graph.n; % Store the graph size.
result.m = graph.m; % Store the edge count.
result.rank = 0; % Use rank zero to denote the full-matrix reference.
result.method = 'Full-SDP-PG'; % Store the full-SDP method name.
result.iterations = last_iteration; % Store the full-SDP iteration count.
result.time = elapsed; % Store the full-SDP runtime.
result.final_gamma = NaN; % Mark gamma as inapplicable.
result.sdp_objective = matrix_sdp_value(graph.L_original,Z); % Store the original-scale full-SDP relaxation value.
result.best_cut = best_cut; % Store the best rounded full-SDP cut.
result.mean_cut = mean_cut; % Store the mean rounded full-SDP cut.
reference_cut = lookup_best_known(best_known,graph.name); % Retrieve the optional best-known cut.
if isfinite(reference_cut) && reference_cut~=0 % Check whether a valid best-known denominator exists.
    result.cut_ratio = best_cut/reference_cut; % Store the ratio to the supplied best-known cut.
else % Handle a missing best-known cut.
    result.cut_ratio = NaN; % Store a missing cut ratio.
end % Finish the full-SDP cut-ratio branch.
result.rel_disagreement = 0; % Set disagreement to zero for a single full-matrix solution.
result.bilinear_sdp_objective = result.sdp_objective; % Use the full-SDP objective as the matching bilinear value.
result.equality_residual = norm(diag(Z)-1,Inf); % Store the full-SDP diagonal-equality residual.
result.multiplier_norm_inf = NaN; % Mark the asymmetric multiplier metric as inapplicable.
result.final_rho = NaN; % Mark the asymmetric feasibility penalty as inapplicable.
result.gamma_candidate = NaN; % Mark the asymmetric gradient-norm metric as inapplicable.
result.gradient_fro = NaN; % Mark the asymmetric gradient metric as inapplicable.
result.gamma_cap_hit = 0; % Record that no asymmetric gamma cap applies.
result.rho_cap_hit = 0; % Record that no asymmetric rho cap applies.
history = array2table(records(1:last_iteration,:),'VariableNames',{'iteration','time','sdp_objective_work','relative_change'}); % Convert the full-SDP history to a table.
if isempty(eigenvalues) % Check whether the PSD factorization unexpectedly returned no positive spectrum.
    warning('The full-SDP factorization returned no positive eigenvalues.'); % Warn about an unexpected numerical degeneracy.
end % Finish the positive-spectrum diagnostic.
end % Finish the full-SDP solver.

function Z = project_elliptope_dykstra(M,max_iter,tol) % Project a symmetric matrix onto the elliptope by Dykstra's algorithm.
X = 0.5*(M+M'); % Symmetrize the projection input.
P = zeros(size(X)); % Initialize the affine-set Dykstra correction.
Q = zeros(size(X)); % Initialize the PSD-set Dykstra correction.
for iteration = 1:max_iter % Start the Dykstra projection loop.
    previous = X; % Store the previous projected matrix.
    affine_input = X+P; % Apply the affine correction.
    affine_projection = affine_input+diag(1-diag(affine_input)); % Project onto matrices with unit diagonal.
    P = affine_input-affine_projection; % Update the affine Dykstra correction.
    psd_input = 0.5*((affine_projection+Q)+(affine_projection+Q)'); % Form the symmetric PSD-projection input.
    [V,D] = eig(psd_input); % Compute the dense eigendecomposition.
    eigenvalues = diag(D); % Extract the eigenvalues.
    eigenvalues(eigenvalues<0) = 0; % Project all negative eigenvalues to zero.
    X = V*diag(eigenvalues)*V'; % Reconstruct the PSD projection.
    X = 0.5*(X+X'); % Remove numerical asymmetry.
    Q = psd_input-X; % Update the PSD Dykstra correction.
    relative_change = norm(X-previous,'fro')/max(1,norm(previous,'fro')); % Compute the Dykstra relative change.
    if relative_change<tol && norm(diag(X)-1)/sqrt(size(X,1))<sqrt(tol) % Check the Dykstra stopping criterion.
        break; % Exit the Dykstra projection loop.
    end % Finish the Dykstra stopping branch.
end % Finish the Dykstra projection loop.
Z = X; % Return the projected matrix.
end % Finish the elliptope projector.

function [U,positive_values] = factorize_psd_matrix(Z) % Build a numerical factor of a PSD matrix.
Z = 0.5*(Z+Z'); % Symmetrize the PSD matrix.
[V,D] = eig(Z); % Compute the dense eigendecomposition.
values = diag(D); % Extract all eigenvalues.
threshold = max(1,max(abs(values)))*1e-10; % Define a numerical positive-eigenvalue threshold.
keep = values>threshold; % Identify positive numerical eigenvalues.
positive_values = values(keep); % Keep positive eigenvalues.
if isempty(positive_values) % Check whether no positive eigenvalues remain.
    U = ones(size(Z,1),1); % Return a deterministic fallback factor.
    U = normalize_rows(U); % Normalize the fallback factor rows.
else % Handle the regular positive-spectrum case.
    U = V(:,keep)*diag(sqrt(positive_values)); % Build a factor satisfying U*U' approximately equal to Z.
    U = normalize_rows(U); % Normalize rows to remove small diagonal projection error before rounding.
end % Finish the positive-spectrum branch.
end % Finish the PSD factorization function.

function save_history_if_requested(history,graph_name,rank_value,method,trial,opt) % Save one method history when requested.
if ~opt.save_histories % Check whether history saving is disabled.
    return; % Return without writing a history file.
end % Finish the disabled-history branch.
filename = safe_filename(sprintf('%s_rank%d_%s_trial%d.csv',graph_name,rank_value,method,trial)); % Build a safe history filename.
writetable(history,fullfile(opt.output_dir,filename)); % Save the method history.
end % Finish the optional history writer.

function filename = safe_filename(filename) % Replace filename characters that are unsafe across operating systems.
unsafe = {'/','\',';',':','*','?','"','<','>','|',' '}; % Define unsafe filename tokens.
for token_index = 1:numel(unsafe) % Loop over unsafe tokens.
    filename = strrep(filename,unsafe{token_index},'_'); % Replace the current unsafe token.
end % Finish the unsafe-token loop.
end % Finish the safe-filename helper.

function summary = aggregate_results(raw_rows) % Aggregate raw trials into one mean-and-standard-deviation row per condition and method.
raw_table = struct2table(raw_rows); % Convert fixed-layout raw results to a table.
keys = cell(height(raw_table),1); % Preallocate grouping keys.
for row_index = 1:height(raw_table) % Loop over raw rows.
    keys{row_index} = sprintf('%s|%d|%d|%s',raw_table.graph_name{row_index},raw_table.n(row_index),raw_table.rank(row_index),raw_table.method{row_index}); % Build the graph-rank-method grouping key.
end % Finish the grouping-key loop.
[unique_keys,first_indices,group_indices] = unique(keys,'stable'); %#ok<ASGLU> % Identify unique graph-rank-method groups.
summary_rows = repmat(summary_row_template(),numel(unique_keys),1); % Preallocate fixed-layout summary rows.
for group_index = 1:numel(unique_keys) % Loop over unique groups.
    members = find(group_indices==group_index); % Find raw rows belonging to the current group.
    first_member = members(1); % Read the first group member.
    summary_rows(group_index).graph_name = raw_table.graph_name{first_member}; % Store the graph name.
    summary_rows(group_index).n = raw_table.n(first_member); % Store the graph size.
    summary_rows(group_index).m = raw_table.m(first_member); % Store the edge count.
    summary_rows(group_index).rank = raw_table.rank(first_member); % Store the factor rank.
    summary_rows(group_index).method = raw_table.method{first_member}; % Store the method name.
    summary_rows(group_index).num_trials = numel(members); % Store the number of available trials.
    [summary_rows(group_index).iterations_mean,summary_rows(group_index).iterations_std] = finite_mean_std(raw_table.iterations(members)); % Aggregate iterations.
    [summary_rows(group_index).time_mean,summary_rows(group_index).time_std] = finite_mean_std(raw_table.time(members)); % Aggregate runtime.
    [summary_rows(group_index).final_gamma_mean,summary_rows(group_index).final_gamma_std] = finite_mean_std(raw_table.final_gamma(members)); % Aggregate final gamma.
    [summary_rows(group_index).sdp_objective_mean,summary_rows(group_index).sdp_objective_std] = finite_mean_std(raw_table.sdp_objective(members)); % Aggregate SDP objective.
    [summary_rows(group_index).best_cut_mean,summary_rows(group_index).best_cut_std] = finite_mean_std(raw_table.best_cut(members)); % Aggregate best rounded cuts.
    [summary_rows(group_index).mean_cut_mean,summary_rows(group_index).mean_cut_std] = finite_mean_std(raw_table.mean_cut(members)); % Aggregate mean rounded cuts.
    [summary_rows(group_index).cut_ratio_mean,summary_rows(group_index).cut_ratio_std] = finite_mean_std(raw_table.cut_ratio(members)); % Aggregate cut ratios.
    [summary_rows(group_index).rel_disagreement_mean,summary_rows(group_index).rel_disagreement_std] = finite_mean_std(raw_table.rel_disagreement(members)); % Aggregate factor disagreement.
    [summary_rows(group_index).bilinear_sdp_objective_mean,summary_rows(group_index).bilinear_sdp_objective_std] = finite_mean_std(raw_table.bilinear_sdp_objective(members)); % Aggregate asymmetric bilinear objectives.
    [summary_rows(group_index).equality_residual_mean,summary_rows(group_index).equality_residual_std] = finite_mean_std(raw_table.equality_residual(members)); % Aggregate diagonal-equality residuals.
    [summary_rows(group_index).multiplier_norm_inf_mean,summary_rows(group_index).multiplier_norm_inf_std] = finite_mean_std(raw_table.multiplier_norm_inf(members)); % Aggregate equality-multiplier norms.
    [summary_rows(group_index).final_rho_mean,summary_rows(group_index).final_rho_std] = finite_mean_std(raw_table.final_rho(members)); % Aggregate feasibility penalties.
    [summary_rows(group_index).gamma_candidate_mean,summary_rows(group_index).gamma_candidate_std] = finite_mean_std(raw_table.gamma_candidate(members)); % Aggregate decreasing gradient-norm candidates.
    [summary_rows(group_index).gradient_fro_mean,summary_rows(group_index).gradient_fro_std] = finite_mean_std(raw_table.gradient_fro(members)); % Aggregate full lifted-gradient Frobenius norms.
    [summary_rows(group_index).gamma_cap_hit_rate,~] = finite_mean_std(raw_table.gamma_cap_hit(members)); % Aggregate the gamma-cap hit rate.
    [summary_rows(group_index).rho_cap_hit_rate,~] = finite_mean_std(raw_table.rho_cap_hit(members)); % Aggregate the rho-cap hit rate.
end % Finish the aggregation loop.
summary = struct2table(summary_rows); % Convert the summary rows to a table.
end % Finish the result aggregator.

function [average,deviation] = finite_mean_std(values) % Compute mean and sample standard deviation over finite values.
values = values(isfinite(values)); % Remove NaN and infinite values.
if isempty(values) % Check whether no finite values remain.
    average = NaN; % Return a missing mean.
    deviation = NaN; % Return a missing standard deviation.
elseif numel(values)==1 % Check whether exactly one finite value remains.
    average = values(1); % Return the single value as the mean.
    deviation = 0; % Return zero standard deviation for one observation.
else % Handle multiple finite values.
    average = mean(values); % Compute the arithmetic mean.
    deviation = std(values,0); % Compute the sample standard deviation.
end % Finish the finite-value branch.
end % Finish the finite mean-and-standard-deviation helper.

function write_configuration(opt,conditions,graph_cache,path) % Save options, conditions, and graph metadata to JSON.
configuration = sanitize_options(opt); % Copy options while removing machine-specific paths.
configuration.conditions = sanitize_conditions(conditions); % Record sanitized graph-and-rank conditions.
cache_keys = keys(graph_cache); % Read all cached graph keys.
metadata = repmat(struct('name','','n',0,'m',0,'scale',0,'weights_nonnegative',false),numel(cache_keys),1); % Preallocate graph metadata.
for index = 1:numel(cache_keys) % Loop over cached graphs.
    graph = graph_cache(cache_keys{index}); % Read the cached graph.
    metadata(index).name = graph.name; % Store the graph name.
    metadata(index).n = graph.n; % Store the graph size.
    metadata(index).m = graph.m; % Store the edge count.
    metadata(index).scale = graph.scale; % Store the optimization Laplacian scale.
    metadata(index).weights_nonnegative = graph.weights_nonnegative; % Store the nonnegative-weight flag.
end % Finish the graph-metadata loop.
configuration.graph_metadata = metadata; % Attach graph metadata to the configuration.
try % Try formatted JSON syntax supported by newer MATLAB releases.
    text = jsonencode(configuration,'PrettyPrint',true); % Encode formatted JSON.
catch % Handle older MATLAB releases.
    text = jsonencode(configuration); % Encode compact JSON.
end % Finish the JSON compatibility branch.
file_identifier = fopen(path,'w'); % Open the configuration file for writing.
if file_identifier<0 % Check whether the configuration file opened successfully.
    error('Cannot write configuration file: %s',path); % Stop on a file-write failure.
end % Finish the configuration-file validation.
cleanup_object = onCleanup(@()fclose(file_identifier)); %#ok<NASGU> % Guarantee that the configuration file is closed.
fprintf(file_identifier,'%s',text); % Write the JSON configuration.
end % Finish the configuration writer.

function clean = sanitize_options(opt) % Remove machine-specific paths before saving reproducibility metadata.
clean = opt; % Copy the option structure.
clean.output_dir = '.'; % Replace the absolute result path by a portable relative path.
if ~isempty(clean.best_known_file) % Check whether a best-known file path was supplied.
    [~,name,extension] = fileparts(clean.best_known_file); % Extract only the portable file name.
    clean.best_known_file = [name extension]; % Store the portable best-known file name.
end % Finish the best-known path branch.
end % Finish the option sanitizer.

function clean = sanitize_conditions(conditions) % Remove directory prefixes from saved graph conditions.
clean = conditions; % Copy the condition cell array.
for index = 1:size(clean,1) % Loop over condition rows.
    source = clean{index,1}; % Read the graph source.
    if ischar(source) && ~strncmp(source,'synthetic:',10) % Check whether the source is a file path.
        [~,name,extension] = fileparts(source); % Extract the portable graph file name.
        clean{index,1} = [name extension]; % Store only the graph file name.
    end % Finish the file-source branch.
end % Finish the condition-sanitization loop.
end % Finish the condition sanitizer.

function row = result_row_template() % Return the fixed raw result field layout.
row = struct('graph_name','','n',0,'m',0,'rank',0,'method','','trial',0,'initialization_seed',0,'rounding_seed',0,'final_gamma_next',NaN,'stop_reason','','iterations',NaN,'time',NaN,'final_gamma',NaN,'final_rho',NaN,'sdp_objective',NaN,'bilinear_sdp_objective',NaN,'best_cut',NaN,'mean_cut',NaN,'cut_ratio',NaN,'rel_disagreement',NaN,'equality_residual',NaN,'multiplier_norm_inf',NaN,'gamma_candidate',NaN,'gradient_fro',NaN,'gamma_cap_hit',0,'rho_cap_hit',0); % Define every raw result field in stable order.
end % Finish the raw result template.

function row = summary_row_template() % Return the fixed summary field layout.
row = struct('graph_name','','n',0,'m',0,'rank',0,'method','','num_trials',0,'iterations_mean',NaN,'iterations_std',NaN,'time_mean',NaN,'time_std',NaN,'final_gamma_mean',NaN,'final_gamma_std',NaN,'final_rho_mean',NaN,'final_rho_std',NaN,'sdp_objective_mean',NaN,'sdp_objective_std',NaN,'bilinear_sdp_objective_mean',NaN,'bilinear_sdp_objective_std',NaN,'best_cut_mean',NaN,'best_cut_std',NaN,'mean_cut_mean',NaN,'mean_cut_std',NaN,'cut_ratio_mean',NaN,'cut_ratio_std',NaN,'rel_disagreement_mean',NaN,'rel_disagreement_std',NaN,'equality_residual_mean',NaN,'equality_residual_std',NaN,'multiplier_norm_inf_mean',NaN,'multiplier_norm_inf_std',NaN,'gamma_candidate_mean',NaN,'gamma_candidate_std',NaN,'gradient_fro_mean',NaN,'gradient_fro_std',NaN,'gamma_cap_hit_rate',NaN,'rho_cap_hit_rate',NaN); % Define every summary field in stable order.
end % Finish the summary row template.

function record = assignment_template() % Return the fixed best-assignment record layout.
record = struct('graph_name','','rank',0,'method','','assignment',[]); % Define every assignment field in stable order.
end % Finish the assignment template.

function record = make_assignment_record(graph_name,rank_value,method,assignment) % Build one best-assignment record.
record = assignment_template(); % Start from the fixed assignment template.
record.graph_name = graph_name; % Store the graph name.
record.rank = rank_value; % Store the rank.
record.method = method; % Store the method name.
record.assignment = assignment; % Store the rounded sign assignment.
end % Finish the assignment-record builder.

function graph = graph_template() % Return the fixed graph structure layout.
graph = struct('name','','source','','n',0,'m',0,'W',sparse(0,0),'L_original',sparse(0,0),'L_work',sparse(0,0),'W_work',sparse(0,0),'scale',1,'gradient_fro',0,'L_work_diagonal',zeros(0,1),'weights_nonnegative',true,'condition_index_first_seen',0); % Define every graph field in stable order.
end % Finish the graph template.
