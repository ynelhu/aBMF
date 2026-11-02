function summary = example_gset() % Run a compact Gset experiment or the included smoke test when external Gset files are absent.
base_directory = fileparts(mfilename('fullpath')); % Locate the directory containing this example file.
data_directory = fullfile(base_directory,'data'); % Build the data-directory path independently of the current MATLAB folder.
[candidate_files,candidate_sizes] = discover_gset_files(data_directory); % Discover user-provided Gset-style graph files and read their declared sizes.
if isempty(candidate_files) % Check whether the user has supplied any external Gset files.
    tiny_path = fullfile(data_directory,'tiny_gset.txt'); % Build the absolute path of the included tiny smoke-test graph.
    warning('No external Gset files were found in %s. Running the included tiny smoke test instead. Add Gset files to this folder and rerun example_gset for the rebuttal experiment.',data_directory); % Explain why the smoke-test configuration is used.
    condition_list = {tiny_path,2}; % Use the included tiny graph with a feasible rank for code validation.
    summary = run_maxcut_benchmark('condition_list',condition_list,'num_trials',1,'rounds',20,'max_iter',100,'tol',1e-6,'epsilon',1e-6,'gamma0','auto','gamma_base',0.25,'rho0',1,'rho_growth',2,'rho_max',1e6,'multiplier_update_period',1,'constraint_tol',1e-6,'full_sdp',true,'max_full_sdp_n',1200,'require_nonnegative_weights',false,'normalize_laplacian',true,'verbose',true,'progress_period',25,'output_dir',fullfile(base_directory,'maxcut_results_tiny_gset')); % Run a short end-to-end smoke test.
    return; % Stop after completing the smoke test.
end % Finish the no-external-data branch.
condition_list = build_rebuttal_conditions(candidate_files,candidate_sizes); % Select compact dimension- and rank-scaling conditions from the available graphs.
summary = run_maxcut_benchmark('condition_list',condition_list,'num_trials',10,'rounds',100,'max_iter',3000,'tol',1e-6,'epsilon',1e-6,'gamma0','auto','gamma_base',0.25,'rho0',1,'rho_growth',2,'rho_max',1e6,'multiplier_update_period',1,'constraint_tol',1e-6,'full_sdp',true,'max_full_sdp_n',1200,'require_nonnegative_weights',false,'normalize_laplacian',true,'verbose',true,'progress_period',100,'output_dir',fullfile(base_directory,'maxcut_results_gset')); % Run the recommended compact rebuttal experiment.
end % Finish the Gset-style rebuttal example.

function [candidate_files,candidate_sizes] = discover_gset_files(data_directory) % Discover external Gset-style files placed in the data directory.
directory_items = dir(data_directory); % List every item in the data directory.
candidate_files = cell(0,1); % Initialize the discovered graph-file paths.
candidate_sizes = zeros(0,1); % Initialize the declared graph sizes.
for item_index = 1:numel(directory_items) % Examine each directory item.
    item = directory_items(item_index); % Read the current directory item.
    if item.isdir % Check whether the current item is a directory.
        continue; % Skip directories.
    end % Finish the directory branch.
    lower_name = lower(item.name); % Convert the file name to lower case for robust filtering.
    if strcmp(lower_name,'readme.md') || strcmp(lower_name,'tiny_gset.txt') || strcmp(lower_name,'best_known_cuts_template.csv') % Check whether the file is bundled documentation or smoke-test data.
        continue; % Skip bundled non-rebuttal files.
    end % Finish the bundled-file filter.
    file_path = fullfile(data_directory,item.name); % Build the absolute candidate-file path.
    declared_n = read_declared_graph_size(file_path); % Read the graph size from the standard Gset header when possible.
    if isnan(declared_n) % Check whether a valid graph size was detected.
        warning('Skipping %s because its first data line is not a standard Gset header of the form "n m".',file_path); % Explain why an unsupported file is skipped.
        continue; % Skip files without a readable standard header.
    end % Finish the invalid-header branch.
    candidate_files{end+1,1} = file_path; %#ok<AGROW> % Append the valid graph-file path.
    candidate_sizes(end+1,1) = declared_n; %#ok<AGROW> % Append the corresponding graph size.
end % Finish the data-directory scan.
[candidate_sizes,sort_order] = sort(candidate_sizes); % Sort valid graphs from smallest to largest.
candidate_files = candidate_files(sort_order); % Apply the same order to the graph-file paths.
end % Finish the external Gset-file discovery function.

function declared_n = read_declared_graph_size(file_path) % Read the vertex count from a standard Gset n-m header.
declared_n = NaN; % Use NaN until a valid header is found.
file_identifier = fopen(file_path,'r'); % Open the candidate graph file.
if file_identifier<0 % Check whether the file opened successfully.
    return; % Leave the size undefined when the file cannot be opened.
end % Finish the file-open branch.
cleanup_object = onCleanup(@()fclose(file_identifier)); %#ok<NASGU> % Guarantee that the candidate file is closed.
while true % Search for the first nonempty noncomment line.
    line = fgetl(file_identifier); % Read the next line.
    if ~ischar(line) % Check whether the end of file was reached.
        return; % Leave the size undefined when no data line exists.
    end % Finish the end-of-file branch.
    line = strtrim(line); % Remove surrounding whitespace.
    if isempty(line) % Check whether the line is empty.
        continue; % Skip empty lines.
    end % Finish the empty-line branch.
    if line(1)=='#' || line(1)=='%' % Check whether the line is a comment.
        continue; % Skip comment lines.
    end % Finish the comment-line branch.
    values = sscanf(line,'%f'); % Parse numeric values from the first data line.
    if numel(values)==2 && values(1)>=2 && values(1)==floor(values(1)) && values(2)>=0 && values(2)==floor(values(2)) % Validate a standard integer n-m header.
        declared_n = double(values(1)); % Store the declared vertex count as double precision.
    end % Finish the valid-header branch.
    return; % Stop after examining the first data line.
end % Finish the header-search loop.
end % Finish the standard-header reader.

function condition_list = build_rebuttal_conditions(candidate_files,candidate_sizes) % Build a compact set of graph-size and rank conditions.
target_sizes = [3000;10000;20000]; % Define the preferred rebuttal graph sizes.
selected_indices = zeros(0,1); % Initialize indices of selected graph files.
for target_index = 1:numel(target_sizes) % Select one available graph near each preferred size.
    [~,nearest_index] = min(abs(candidate_sizes-target_sizes(target_index))); % Find the available graph closest to the current target size.
    if ~any(selected_indices==nearest_index) % Check whether this graph has already been selected.
        selected_indices(end+1,1) = nearest_index; %#ok<AGROW> % Add the unique nearest graph.
    end % Finish the duplicate-selection branch.
end % Finish the target-size selection loop.
if isempty(selected_indices) % Guard against an unexpected empty selection.
    selected_indices = numel(candidate_files); % Select the largest available graph as a fallback.
end % Finish the empty-selection fallback.
selected_indices = sort(selected_indices); % Order selected graphs from smaller to larger size.
condition_list = cell(0,2); % Initialize the graph-rank condition list.
for selected_position = 1:numel(selected_indices) % Add the rank-50 dimension-scaling conditions.
    graph_index = selected_indices(selected_position); % Read the selected graph index.
    rank_value = min(50,candidate_sizes(graph_index)); % Ensure that the requested rank does not exceed the graph dimension.
    condition_list(end+1,:) = {candidate_files{graph_index},rank_value}; %#ok<AGROW> % Add the selected graph with rank fifty when feasible.
end % Finish the dimension-scaling condition loop.
[~,largest_selected_position] = max(candidate_sizes(selected_indices)); % Locate the largest selected graph.
largest_graph_index = selected_indices(largest_selected_position); % Read the largest selected graph index.
high_rank = min(100,candidate_sizes(largest_graph_index)); % Define the rank-scaling stress-test rank without exceeding the dimension.
base_rank = min(50,candidate_sizes(largest_graph_index)); % Define the base rank used for the largest graph.
if high_rank>base_rank % Check whether the high-rank condition differs from the base condition.
    condition_list(end+1,:) = {candidate_files{largest_graph_index},high_rank}; %#ok<AGROW> % Add the high-rank stress-test condition on the largest graph.
end % Finish the high-rank condition branch.
fprintf('Selected Gset rebuttal conditions:\n'); % Introduce the automatically selected conditions.
for condition_index = 1:size(condition_list,1) % Display each selected graph-rank pair.
    fprintf('  %s, rank = %d\n',condition_list{condition_index,1},condition_list{condition_index,2}); % Print the selected file and rank.
end % Finish the selected-condition display loop.
end % Finish the compact rebuttal-condition builder.
