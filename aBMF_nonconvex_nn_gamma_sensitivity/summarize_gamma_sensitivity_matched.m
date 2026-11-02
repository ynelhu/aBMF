function matched_summary = summarize_gamma_sensitivity_matched(raw_results,gamma_multipliers) % Compute matched changes relative to the c=1 fixed-gamma runs.
base_multiplier = 1; % Use c=1 as the controlled reference condition.
matched_summary = repmat(matched_summary_template(),numel(gamma_multipliers),1); % Allocate one matched-summary row per multiplier.
for multiplier_index = 1:numel(gamma_multipliers) % Process every requested multiplier.
    multiplier_value = gamma_multipliers(multiplier_index); % Read the current multiplier.
    objective_deltas = zeros(0,1); % Allocate matched surrogate-objective changes.
    margin_deltas = zeros(0,1); % Allocate matched signed-margin changes.
    disagreement_ratios = zeros(0,1); % Allocate matched disagreement ratios.
    iteration_ratios = zeros(0,1); % Allocate matched iteration-count ratios.
    time_ratios = zeros(0,1); % Allocate matched runtime ratios.
    success_differences = zeros(0,1); % Allocate matched success-indicator differences.
    for row_index = 1:numel(raw_results) % Search every raw-result row for the current multiplier.
        if abs(raw_results(row_index).gamma_multiplier-multiplier_value) > 1e-12*max(1,abs(multiplier_value)) % Skip rows belonging to another multiplier.
            continue; % Continue to the next raw-result row.
        end % Finish the current-multiplier check.
        base_index = find_matching_base_row(raw_results,raw_results(row_index),base_multiplier); % Locate the c=1 run with identical sample, epsilon, rank, and trial.
        if base_index <= 0 % Check whether the matched base run exists.
            error('A matched c=1 reference run is missing.'); % Stop because the controlled comparison is incomplete.
        end % Finish the matched-base check.
        objective_deltas(end+1,1) = raw_results(row_index).surrogate_objective-raw_results(base_index).surrogate_objective; % Store the matched objective change.
        margin_deltas(end+1,1) = raw_results(row_index).margin_surrogate-raw_results(base_index).margin_surrogate; % Store the matched signed-margin change.
        disagreement_ratios(end+1,1) = raw_results(row_index).rel_disagreement/max(raw_results(base_index).rel_disagreement,1e-12); % Store the matched disagreement ratio.
        iteration_ratios(end+1,1) = raw_results(row_index).iterations/max(raw_results(base_index).iterations,1); % Store the matched iteration-count ratio.
        time_ratios(end+1,1) = raw_results(row_index).time/max(raw_results(base_index).time,1e-12); % Store the matched runtime ratio.
        success_differences(end+1,1) = raw_results(row_index).success-raw_results(base_index).success; % Store the matched success-indicator difference.
    end % Finish searching raw-result rows for the current multiplier.
    matched_summary(multiplier_index).gamma_multiplier = multiplier_value; % Store the controlled multiplier.
    matched_summary(multiplier_index).num_pairs = numel(objective_deltas); % Store the number of matched comparisons.
    matched_summary(multiplier_index).objective_delta_mean = mean(objective_deltas); % Compute the mean matched objective change.
    matched_summary(multiplier_index).objective_delta_std = std(objective_deltas,0); % Compute the matched objective-change standard deviation.
    matched_summary(multiplier_index).margin_delta_mean = mean(margin_deltas); % Compute the mean matched signed-margin change.
    matched_summary(multiplier_index).margin_delta_std = std(margin_deltas,0); % Compute the matched signed-margin standard deviation.
    matched_summary(multiplier_index).disagreement_ratio_mean = mean(disagreement_ratios); % Compute the mean matched disagreement ratio.
    matched_summary(multiplier_index).disagreement_ratio_std = std(disagreement_ratios,0); % Compute the disagreement-ratio standard deviation.
    matched_summary(multiplier_index).iteration_ratio_mean = mean(iteration_ratios); % Compute the mean matched iteration-count ratio.
    matched_summary(multiplier_index).iteration_ratio_std = std(iteration_ratios,0); % Compute the iteration-ratio standard deviation.
    matched_summary(multiplier_index).time_ratio_mean = mean(time_ratios); % Compute the mean matched runtime ratio.
    matched_summary(multiplier_index).time_ratio_std = std(time_ratios,0); % Compute the runtime-ratio standard deviation.
    matched_summary(multiplier_index).success_difference_mean = mean(success_differences); % Compute the mean matched change in diagnostic success.
end % Finish processing every multiplier.
end % Finish the matched sensitivity summary.

function base_index = find_matching_base_row(raw_results,current_row,base_multiplier) % Locate the c=1 row matching all non-gamma variables.
base_index = 0; % Initialize the missing-match marker.
for candidate_index = 1:numel(raw_results) % Search every raw-result row.
    candidate = raw_results(candidate_index); % Read the current candidate row.
    same_sample = candidate.sample_index == current_row.sample_index; % Test the selected-sample index.
    same_epsilon = abs(candidate.epsilon-current_row.epsilon) <= 1e-12*max(1,abs(current_row.epsilon)); % Test the perturbation radius.
    same_rank = candidate.rank == current_row.rank; % Test the factor rank.
    same_trial = candidate.trial == current_row.trial; % Test the matched random trial.
    same_multiplier = abs(candidate.gamma_multiplier-base_multiplier) <= 1e-12*max(1,abs(base_multiplier)); % Test the c=1 multiplier.
    if same_sample && same_epsilon && same_rank && same_trial && same_multiplier % Check whether every controlled variable matches.
        base_index = candidate_index; % Store the matched c=1 row index.
        return; % Finish the search immediately.
    end % Finish the complete-match check.
end % Finish searching all raw-result rows.
end % Finish the matched-base-row search.

function row = matched_summary_template() % Create one fixed-layout matched-summary structure.
row = struct('gamma_multiplier',NaN,'num_pairs',0,'objective_delta_mean',NaN,'objective_delta_std',NaN,'margin_delta_mean',NaN,'margin_delta_std',NaN,'disagreement_ratio_mean',NaN,'disagreement_ratio_std',NaN,'iteration_ratio_mean',NaN,'iteration_ratio_std',NaN,'time_ratio_mean',NaN,'time_ratio_std',NaN,'success_difference_mean',NaN); % Define all matched-summary fields in a stable order.
end % Finish the matched-summary template.
