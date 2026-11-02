function display_gamma_sensitivity_aggregate(aggregate,matched_summary) % Display compact multiplier-level and matched summaries.
fprintf('\nOverall fixed-gamma multiplier sensitivity:\n'); % Print the overall-summary heading.
fprintf('c       runs    gamma_mean    objective      disagreement   iterations   time(s)   success\n'); % Print the overall-summary column labels.
for row_index = 1:numel(aggregate) % Display every multiplier-level aggregate row.
    row = aggregate(row_index); % Read the current aggregate row.
    fprintf('%-7.4g %-7d %-13.5g %-14.6g %-14.4e %-12.3f %-9.3f %.3f\n',row.gamma_multiplier,row.num_runs,row.gamma_value_mean,row.surrogate_objective_mean,row.rel_disagreement_mean,row.iterations_mean,row.time_mean,row.success_rate); % Display the current aggregate values.
end % Finish displaying the overall aggregate.
fprintf('\nMatched changes relative to c=1:\n'); % Print the matched-summary heading.
fprintf('c       pairs   objective_delta   disagreement_ratio   iteration_ratio   time_ratio   success_delta\n'); % Print the matched-summary column labels.
for row_index = 1:numel(matched_summary) % Display every matched-summary row.
    row = matched_summary(row_index); % Read the current matched-summary row.
    fprintf('%-7.4g %-7d %-17.6g %-20.5g %-17.5g %-12.5g %.5g\n',row.gamma_multiplier,row.num_pairs,row.objective_delta_mean,row.disagreement_ratio_mean,row.iteration_ratio_mean,row.time_ratio_mean,row.success_difference_mean); % Display the current matched values.
end % Finish displaying the matched summary.
end % Finish the compact sensitivity display.
