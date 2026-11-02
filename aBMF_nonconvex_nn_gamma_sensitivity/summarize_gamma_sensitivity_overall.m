function aggregate = summarize_gamma_sensitivity_overall(raw_results,gamma_multipliers) % Aggregate the complete controlled experiment by gamma multiplier.
aggregate = repmat(overall_summary_template(),numel(gamma_multipliers),1); % Allocate one aggregate row per multiplier.
for multiplier_index = 1:numel(gamma_multipliers) % Aggregate every requested multiplier.
    multiplier_value = gamma_multipliers(multiplier_index); % Read the current multiplier.
    selected = find(abs(extract_all(raw_results,'gamma_multiplier')-multiplier_value) <= 1e-12*max(1,abs(multiplier_value))); % Locate all raw runs using the current multiplier.
    aggregate(multiplier_index).gamma_multiplier = multiplier_value; % Store the controlled multiplier.
    aggregate(multiplier_index).num_runs = numel(selected); % Store the number of complete runs.
    aggregate(multiplier_index).gamma_reference_mean = mean(extract_selected(raw_results,selected,'gamma_reference')); % Compute the mean gamma reference.
    aggregate(multiplier_index).gamma_reference_std = std(extract_selected(raw_results,selected,'gamma_reference'),0); % Compute the gamma-reference standard deviation.
    aggregate(multiplier_index).gamma_value_mean = mean(extract_selected(raw_results,selected,'gamma_value')); % Compute the mean actual fixed gamma.
    aggregate(multiplier_index).gamma_value_std = std(extract_selected(raw_results,selected,'gamma_value'),0); % Compute the actual-gamma standard deviation.
    aggregate(multiplier_index).iterations_mean = mean(extract_selected(raw_results,selected,'iterations')); % Compute the mean iteration count.
    aggregate(multiplier_index).iterations_std = std(extract_selected(raw_results,selected,'iterations'),0); % Compute the iteration-count standard deviation.
    aggregate(multiplier_index).time_mean = mean(extract_selected(raw_results,selected,'time')); % Compute the mean runtime.
    aggregate(multiplier_index).time_std = std(extract_selected(raw_results,selected,'time'),0); % Compute the runtime standard deviation.
    aggregate(multiplier_index).surrogate_objective_mean = mean(extract_selected(raw_results,selected,'surrogate_objective')); % Compute the mean surrogate objective.
    aggregate(multiplier_index).surrogate_objective_std = std(extract_selected(raw_results,selected,'surrogate_objective'),0); % Compute the surrogate-objective standard deviation.
    aggregate(multiplier_index).margin_surrogate_mean = mean(extract_selected(raw_results,selected,'margin_surrogate')); % Compute the mean signed margin surrogate.
    aggregate(multiplier_index).margin_surrogate_std = std(extract_selected(raw_results,selected,'margin_surrogate'),0); % Compute the signed-margin standard deviation.
    aggregate(multiplier_index).affine_residual_mean = mean(extract_selected(raw_results,selected,'affine_residual')); % Compute the mean affine residual.
    aggregate(multiplier_index).affine_residual_std = std(extract_selected(raw_results,selected,'affine_residual'),0); % Compute the affine-residual standard deviation.
    aggregate(multiplier_index).inequality_residual_mean = mean(extract_selected(raw_results,selected,'inequality_residual')); % Compute the mean inequality residual.
    aggregate(multiplier_index).inequality_residual_std = std(extract_selected(raw_results,selected,'inequality_residual'),0); % Compute the inequality-residual standard deviation.
    aggregate(multiplier_index).complementarity_residual_mean = mean(extract_selected(raw_results,selected,'complementarity_residual')); % Compute the mean complementarity residual.
    aggregate(multiplier_index).complementarity_residual_std = std(extract_selected(raw_results,selected,'complementarity_residual'),0); % Compute the complementarity-residual standard deviation.
    aggregate(multiplier_index).rel_disagreement_mean = mean(extract_selected(raw_results,selected,'rel_disagreement')); % Compute the mean relative factor disagreement.
    aggregate(multiplier_index).rel_disagreement_std = std(extract_selected(raw_results,selected,'rel_disagreement'),0); % Compute the disagreement standard deviation.
    aggregate(multiplier_index).psd_violation_mean = mean(extract_selected(raw_results,selected,'psd_violation')); % Compute the mean PSD violation.
    aggregate(multiplier_index).psd_violation_std = std(extract_selected(raw_results,selected,'psd_violation'),0); % Compute the PSD-violation standard deviation.
    aggregate(multiplier_index).success_rate = mean(extract_selected(raw_results,selected,'success')); % Compute the common diagnostic success rate.
end % Finish aggregating every multiplier.
end % Finish the multiplier-level aggregate summary.

function values = extract_all(raw_results,field_name) % Extract one numeric field from every raw-result row.
values = zeros(numel(raw_results),1); % Allocate the complete numeric vector.
for row_index = 1:numel(raw_results) % Read every raw-result row.
    values(row_index) = raw_results(row_index).(field_name); % Store the requested numeric field.
end % Finish extracting the complete field.
end % Finish the complete-field extractor.

function values = extract_selected(raw_results,selected,field_name) % Extract one numeric field from selected raw-result rows.
values = zeros(numel(selected),1); % Allocate the selected numeric vector.
for value_index = 1:numel(selected) % Read every selected row.
    values(value_index) = raw_results(selected(value_index)).(field_name); % Store the requested numeric field.
end % Finish extracting the selected field.
end % Finish the selected-field extractor.

function row = overall_summary_template() % Create one fixed-layout multiplier-level summary structure.
row = struct('gamma_multiplier',NaN,'num_runs',0,'gamma_reference_mean',NaN,'gamma_reference_std',NaN,'gamma_value_mean',NaN,'gamma_value_std',NaN,'iterations_mean',NaN,'iterations_std',NaN,'time_mean',NaN,'time_std',NaN,'surrogate_objective_mean',NaN,'surrogate_objective_std',NaN,'margin_surrogate_mean',NaN,'margin_surrogate_std',NaN,'affine_residual_mean',NaN,'affine_residual_std',NaN,'inequality_residual_mean',NaN,'inequality_residual_std',NaN,'complementarity_residual_mean',NaN,'complementarity_residual_std',NaN,'rel_disagreement_mean',NaN,'rel_disagreement_std',NaN,'psd_violation_mean',NaN,'psd_violation_std',NaN,'success_rate',NaN); % Define all multiplier-level summary fields in a stable order.
end % Finish the multiplier-level summary template.
