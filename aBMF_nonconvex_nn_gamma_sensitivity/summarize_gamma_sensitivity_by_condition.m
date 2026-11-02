function summary = summarize_gamma_sensitivity_by_condition(raw_results) % Aggregate gamma-sensitivity results by sample, epsilon, rank, and multiplier.
keys = zeros(numel(raw_results),4); % Allocate the numeric grouping-key matrix.
for row_index = 1:numel(raw_results) % Construct one grouping key per raw-result row.
    keys(row_index,:) = [raw_results(row_index).sample_index,raw_results(row_index).epsilon,raw_results(row_index).rank,raw_results(row_index).gamma_multiplier]; % Store the current condition key.
end % Finish constructing grouping keys.
[unique_keys,~,group_ids] = unique(keys,'rows'); % Identify all distinct controlled conditions.
summary = repmat(condition_summary_template(),size(unique_keys,1),1); % Allocate the condition-level summary structure.
for group_index = 1:size(unique_keys,1) % Aggregate every controlled condition.
    selected = find(group_ids == group_index); % Locate all raw runs belonging to this condition.
    summary(group_index).sample_index = unique_keys(group_index,1); % Store the selected test-set index.
    summary(group_index).epsilon = unique_keys(group_index,2); % Store the perturbation radius.
    summary(group_index).rank = unique_keys(group_index,3); % Store the factor rank.
    summary(group_index).gamma_multiplier = unique_keys(group_index,4); % Store the controlled multiplier.
    summary(group_index).num_trials = numel(selected); % Store the number of matched trials.
    summary(group_index).gamma_reference_mean = mean(extract_field(raw_results,selected,'gamma_reference')); % Compute the mean gamma reference.
    summary(group_index).gamma_reference_std = std(extract_field(raw_results,selected,'gamma_reference'),0); % Compute the gamma-reference standard deviation.
    summary(group_index).gamma_value_mean = mean(extract_field(raw_results,selected,'gamma_value')); % Compute the mean fixed gamma.
    summary(group_index).gamma_value_std = std(extract_field(raw_results,selected,'gamma_value'),0); % Compute the fixed-gamma standard deviation.
    summary(group_index).iterations_mean = mean(extract_field(raw_results,selected,'iterations')); % Compute the mean iteration count.
    summary(group_index).iterations_std = std(extract_field(raw_results,selected,'iterations'),0); % Compute the iteration-count standard deviation.
    summary(group_index).time_mean = mean(extract_field(raw_results,selected,'time')); % Compute the mean runtime.
    summary(group_index).time_std = std(extract_field(raw_results,selected,'time'),0); % Compute the runtime standard deviation.
    summary(group_index).surrogate_objective_mean = mean(extract_field(raw_results,selected,'surrogate_objective')); % Compute the mean surrogate objective.
    summary(group_index).surrogate_objective_std = std(extract_field(raw_results,selected,'surrogate_objective'),0); % Compute the surrogate-objective standard deviation.
    summary(group_index).margin_surrogate_mean = mean(extract_field(raw_results,selected,'margin_surrogate')); % Compute the mean signed margin surrogate.
    summary(group_index).margin_surrogate_std = std(extract_field(raw_results,selected,'margin_surrogate'),0); % Compute the signed-margin standard deviation.
    summary(group_index).affine_residual_mean = mean(extract_field(raw_results,selected,'affine_residual')); % Compute the mean affine residual.
    summary(group_index).affine_residual_std = std(extract_field(raw_results,selected,'affine_residual'),0); % Compute the affine-residual standard deviation.
    summary(group_index).inequality_residual_mean = mean(extract_field(raw_results,selected,'inequality_residual')); % Compute the mean inequality residual.
    summary(group_index).inequality_residual_std = std(extract_field(raw_results,selected,'inequality_residual'),0); % Compute the inequality-residual standard deviation.
    summary(group_index).complementarity_residual_mean = mean(extract_field(raw_results,selected,'complementarity_residual')); % Compute the mean complementarity residual.
    summary(group_index).complementarity_residual_std = std(extract_field(raw_results,selected,'complementarity_residual'),0); % Compute the complementarity-residual standard deviation.
    summary(group_index).rel_disagreement_mean = mean(extract_field(raw_results,selected,'rel_disagreement')); % Compute the mean relative factor disagreement.
    summary(group_index).rel_disagreement_std = std(extract_field(raw_results,selected,'rel_disagreement'),0); % Compute the disagreement standard deviation.
    summary(group_index).psd_violation_mean = mean(extract_field(raw_results,selected,'psd_violation')); % Compute the mean PSD violation.
    summary(group_index).psd_violation_std = std(extract_field(raw_results,selected,'psd_violation'),0); % Compute the PSD-violation standard deviation.
    summary(group_index).success_rate = mean(extract_field(raw_results,selected,'success')); % Compute the common diagnostic success rate.
end % Finish aggregating all controlled conditions.
end % Finish the condition-level sensitivity summary.

function values = extract_field(raw_results,selected,field_name) % Extract one numeric raw-result field for selected rows.
values = zeros(numel(selected),1); % Allocate the extracted numeric vector.
for value_index = 1:numel(selected) % Read every selected row.
    values(value_index) = raw_results(selected(value_index)).(field_name); % Store the requested numeric field.
end % Finish extracting the requested field.
end % Finish the numeric-field extractor.

function row = condition_summary_template() % Create one fixed-layout condition-summary structure.
row = struct('sample_index',0,'epsilon',NaN,'rank',0,'gamma_multiplier',NaN,'num_trials',0,'gamma_reference_mean',NaN,'gamma_reference_std',NaN,'gamma_value_mean',NaN,'gamma_value_std',NaN,'iterations_mean',NaN,'iterations_std',NaN,'time_mean',NaN,'time_std',NaN,'surrogate_objective_mean',NaN,'surrogate_objective_std',NaN,'margin_surrogate_mean',NaN,'margin_surrogate_std',NaN,'affine_residual_mean',NaN,'affine_residual_std',NaN,'inequality_residual_mean',NaN,'inequality_residual_std',NaN,'complementarity_residual_mean',NaN,'complementarity_residual_std',NaN,'rel_disagreement_mean',NaN,'rel_disagreement_std',NaN,'psd_violation_mean',NaN,'psd_violation_std',NaN,'success_rate',NaN); % Define all condition-summary fields in a stable order.
end % Finish the condition-summary template.
