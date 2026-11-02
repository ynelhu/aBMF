function summary = summarize_nc_results(raw_results) % Aggregate raw results by sample, epsilon, rank, and method.
summary = repmat(summary_row_template(),0,1); % Allocate an empty fixed-layout summary structure.
group_assignment = zeros(numel(raw_results),1); % Allocate the summary-group assignment of every raw row.
for raw_index = 1:numel(raw_results) % Discover all unique experimental groups.
    matched_group = 0; % Initialize the matched summary-group index.
    for summary_index = 1:numel(summary) % Compare the current raw row against existing groups.
        same_sample = summary(summary_index).sample_index == raw_results(raw_index).sample_index; % Compare sample indices.
        same_epsilon = summary(summary_index).epsilon == raw_results(raw_index).epsilon; % Compare perturbation radii.
        same_rank = summary(summary_index).rank == raw_results(raw_index).rank; % Compare factor ranks.
        same_method = strcmp(summary(summary_index).method,raw_results(raw_index).method); % Compare method names.
        if same_sample && same_epsilon && same_rank && same_method % Check whether all grouping fields match.
            matched_group = summary_index; % Record the matching group.
            break; % Stop searching for an existing group.
        end % Finish the group-match check.
    end % Finish the existing-group search.
    if matched_group == 0 % Check whether a new group is required.
        new_row = summary_row_template(); % Create one fixed-layout summary row.
        new_row.sample_index = raw_results(raw_index).sample_index; % Store the group sample index.
        new_row.epsilon = raw_results(raw_index).epsilon; % Store the group perturbation radius.
        new_row.rank = raw_results(raw_index).rank; % Store the group factor rank.
        new_row.method = raw_results(raw_index).method; % Store the group method name.
        summary(end+1,1) = new_row; % Append the new summary group.
        matched_group = numel(summary); % Record the appended group index.
    end % Finish new-group creation.
    group_assignment(raw_index) = matched_group; % Assign the raw row to its summary group.
end % Finish discovering summary groups.
for summary_index = 1:numel(summary) % Compute statistics for every summary group.
    selected = group_assignment == summary_index; % Select all raw rows belonging to the current group.
    summary(summary_index).num_trials = sum(selected); % Count the available trials.
    [summary(summary_index).iterations_mean,summary(summary_index).iterations_std] = finite_mean_std([raw_results(selected).iterations]); % Aggregate iteration counts.
    [summary(summary_index).time_mean,summary(summary_index).time_std] = finite_mean_std([raw_results(selected).time]); % Aggregate wall-clock times.
    [summary(summary_index).final_gamma_mean,summary(summary_index).final_gamma_std] = finite_mean_std([raw_results(selected).final_gamma]); % Aggregate final penalties.
    [summary(summary_index).final_gamma_hat_mean,summary(summary_index).final_gamma_hat_std] = finite_mean_std([raw_results(selected).final_gamma_hat]); % Aggregate direct penalty candidates.
    [summary(summary_index).surrogate_objective_mean,summary(summary_index).surrogate_objective_std] = finite_mean_std([raw_results(selected).surrogate_objective]); % Aggregate surrogate objectives.
    [summary(summary_index).margin_surrogate_mean,summary(summary_index).margin_surrogate_std] = finite_mean_std([raw_results(selected).margin_surrogate]); % Aggregate margin surrogates.
    [summary(summary_index).affine_residual_mean,summary(summary_index).affine_residual_std] = finite_mean_std([raw_results(selected).affine_residual]); % Aggregate affine residuals.
    [summary(summary_index).inequality_residual_mean,summary(summary_index).inequality_residual_std] = finite_mean_std([raw_results(selected).inequality_residual]); % Aggregate inequality residuals.
    [summary(summary_index).complementarity_residual_mean,summary(summary_index).complementarity_residual_std] = finite_mean_std([raw_results(selected).complementarity_residual]); % Aggregate complementarity residuals.
    [summary(summary_index).rel_disagreement_mean,summary(summary_index).rel_disagreement_std] = finite_mean_std([raw_results(selected).rel_disagreement]); % Aggregate factor disagreements.
    [summary(summary_index).psd_violation_mean,summary(summary_index).psd_violation_std] = finite_mean_std([raw_results(selected).psd_violation]); % Aggregate PSD violations.
    [summary(summary_index).success_rate,~] = finite_mean_std([raw_results(selected).success]); % Compute the common diagnostic success rate.
end % Finish computing group statistics.
end % Finish result summarization.

function aggregate = aggregate_rebuttal_results(raw_results) % Aggregate all selected conditions into one compact method-level rebuttal table.
method_list = unique_method_list(raw_results); % Obtain method names in first-appearance order.
aggregate = repmat(aggregate_row_template(),numel(method_list),1); % Allocate one aggregate row per method.
for method_index = 1:numel(method_list) % Aggregate every method across all conditions and trials.
    selected = strcmp({raw_results.method},method_list{method_index}); % Select all rows for the current method.
    aggregate(method_index).method = method_list{method_index}; % Store the method name.
    aggregate(method_index).num_runs = sum(selected); % Count all available runs.
    [aggregate(method_index).time_mean,aggregate(method_index).time_std] = finite_mean_std([raw_results(selected).time]); % Aggregate wall-clock time.
    [aggregate(method_index).final_gamma_mean,aggregate(method_index).final_gamma_std] = finite_mean_std([raw_results(selected).final_gamma]); % Aggregate final gamma.
    [aggregate(method_index).surrogate_objective_mean,aggregate(method_index).surrogate_objective_std] = finite_mean_std([raw_results(selected).surrogate_objective]); % Aggregate the nonconvex surrogate objective.
    [aggregate(method_index).margin_surrogate_mean,aggregate(method_index).margin_surrogate_std] = finite_mean_std([raw_results(selected).margin_surrogate]); % Aggregate the margin surrogate.
    [aggregate(method_index).affine_residual_mean,aggregate(method_index).affine_residual_std] = finite_mean_std([raw_results(selected).affine_residual]); % Aggregate affine residuals.
    [aggregate(method_index).inequality_residual_mean,aggregate(method_index).inequality_residual_std] = finite_mean_std([raw_results(selected).inequality_residual]); % Aggregate inequality residuals.
    [aggregate(method_index).complementarity_residual_mean,aggregate(method_index).complementarity_residual_std] = finite_mean_std([raw_results(selected).complementarity_residual]); % Aggregate complementarity residuals.
    [aggregate(method_index).rel_disagreement_mean,aggregate(method_index).rel_disagreement_std] = finite_mean_std([raw_results(selected).rel_disagreement]); % Aggregate factor disagreements.
    [aggregate(method_index).psd_violation_mean,aggregate(method_index).psd_violation_std] = finite_mean_std([raw_results(selected).psd_violation]); % Aggregate PSD violations.
    [aggregate(method_index).success_rate,~] = finite_mean_std([raw_results(selected).success]); % Aggregate diagnostic success flags.
end % Finish method-level aggregation.
end % Finish the compact rebuttal aggregation.

function method_list = unique_method_list(raw_results) % Obtain method names in stable first-appearance order.
method_list = cell(0,1); % Allocate an empty method-name list.
for raw_index = 1:numel(raw_results) % Inspect every raw result row.
    current_method = raw_results(raw_index).method; % Read the current method name.
    if ~any(strcmp(method_list,current_method)) % Check whether the method has already been recorded.
        method_list{end+1,1} = current_method; % Append the newly encountered method name.
    end % Finish the method-existence check.
end % Finish the raw-result scan.
end % Finish stable method-name extraction.

function [mean_value,std_value] = finite_mean_std(value_vector) % Compute mean and sample standard deviation after removing nonfinite values.
value_vector = value_vector(isfinite(value_vector)); % Retain only finite observations.
if isempty(value_vector) % Handle a metric with no finite observations.
    mean_value = NaN; % Mark the mean as unavailable.
    std_value = NaN; % Mark the standard deviation as unavailable.
elseif numel(value_vector) == 1 % Handle a single finite observation.
    mean_value = value_vector(1); % Return the observation as the mean.
    std_value = 0; % Define the displayed standard deviation as zero.
else % Handle two or more finite observations.
    mean_value = sum(value_vector)/numel(value_vector); % Compute the arithmetic mean.
    std_value = sqrt(sum((value_vector-mean_value).^2)/(numel(value_vector)-1)); % Compute the sample standard deviation.
end % Finish the observation-count check.
end % Finish finite statistics.

function row = summary_row_template() % Create one fixed-layout condition-level summary row.
row = struct('sample_index',0,'epsilon',NaN,'rank',0,'method','','num_trials',0,'iterations_mean',NaN,'iterations_std',NaN,'time_mean',NaN,'time_std',NaN,'final_gamma_mean',NaN,'final_gamma_std',NaN,'final_gamma_hat_mean',NaN,'final_gamma_hat_std',NaN,'surrogate_objective_mean',NaN,'surrogate_objective_std',NaN,'margin_surrogate_mean',NaN,'margin_surrogate_std',NaN,'affine_residual_mean',NaN,'affine_residual_std',NaN,'inequality_residual_mean',NaN,'inequality_residual_std',NaN,'complementarity_residual_mean',NaN,'complementarity_residual_std',NaN,'rel_disagreement_mean',NaN,'rel_disagreement_std',NaN,'psd_violation_mean',NaN,'psd_violation_std',NaN,'success_rate',NaN); % Define all condition-level summary fields.
end % Finish the summary-row template.

function row = aggregate_row_template() % Create one fixed-layout method-level aggregate row.
row = struct('method','','num_runs',0,'time_mean',NaN,'time_std',NaN,'final_gamma_mean',NaN,'final_gamma_std',NaN,'surrogate_objective_mean',NaN,'surrogate_objective_std',NaN,'margin_surrogate_mean',NaN,'margin_surrogate_std',NaN,'affine_residual_mean',NaN,'affine_residual_std',NaN,'inequality_residual_mean',NaN,'inequality_residual_std',NaN,'complementarity_residual_mean',NaN,'complementarity_residual_std',NaN,'rel_disagreement_mean',NaN,'rel_disagreement_std',NaN,'psd_violation_mean',NaN,'psd_violation_std',NaN,'success_rate',NaN); % Define all method-level aggregate fields.
end % Finish the aggregate-row template.
