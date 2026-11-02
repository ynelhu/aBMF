function summary = example_gset_quick() % Run one short Gset condition to verify progress and runtime before the full benchmark.
base_directory = fileparts(mfilename('fullpath')); % Locate the project root directory.
data_directory = fullfile(base_directory,'data'); % Build the graph-data directory path.
candidate_names = {'G67.txt','G55.txt','G20.txt'}; % Prefer the signed G67 instance so the quick test validates signed-weight support.
graph_path = ''; % Initialize the selected graph path.
for candidate_index = 1:numel(candidate_names) % Search for the first available preferred graph.
    candidate_path = fullfile(data_directory,candidate_names{candidate_index}); % Build the current candidate path.
    if exist(candidate_path,'file')==2 % Check whether the candidate graph file exists.
        graph_path = candidate_path; % Select the available candidate graph.
        break; % Stop after selecting one graph.
    end % Finish the graph-existence branch.
end % Finish the candidate graph loop.
if isempty(graph_path) % Check whether none of the preferred external graphs was found.
    graph_path = fullfile(data_directory,'tiny_gset.txt'); % Fall back to the included tiny graph.
    rank_value = 2; % Use a feasible rank for the tiny graph.
else % Handle an available external Gset graph.
    rank_value = 50; % Use the rebuttal base rank for the quick external-graph test.
end % Finish the graph-selection branch.
fprintf('Quick Max-Cut test: %s, rank=%d.
',graph_path,rank_value); % Print the selected quick-test condition.
condition_list = {graph_path,rank_value}; % Build the single quick-test condition.
summary = run_maxcut_benchmark('condition_list',condition_list,'num_trials',1,'rounds',20,'max_iter',100,'tol',1e-5,'epsilon',1e-6,'gamma0','auto','gamma_base',0.25,'rho0',1,'rho_growth',2,'rho_max',1e6,'multiplier_update_period',1,'constraint_tol',1e-5,'full_sdp',false,'require_nonnegative_weights',false,'normalize_laplacian',true,'verbose',true,'progress_period',10,'output_dir',fullfile(base_directory,'maxcut_results_quick')); % Run one short progress-visible smoke test.
end % Finish the quick Gset example.
