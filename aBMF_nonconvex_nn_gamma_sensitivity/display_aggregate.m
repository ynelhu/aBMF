function display_aggregate(aggregate) % Display the compact method-level results in the MATLAB command window.
fprintf('\nCompact aggregate over all selected conditions:\n'); % Print the aggregate heading.
for row_index = 1:numel(aggregate) % Display every method-level aggregate row.
    row = aggregate(row_index); % Read the current aggregate row.
    fprintf('%-12s runs=%-3d obj=%10.4g comp=%9.3e gap=%9.3e psd=%9.3e time=%8.3g success=%.3f\n',row.method,row.num_runs,row.surrogate_objective_mean,row.complementarity_residual_mean,row.rel_disagreement_mean,row.psd_violation_mean,row.time_mean,row.success_rate); % Display the main rebuttal diagnostics.
end % Finish displaying aggregate rows.
end % Finish command-window aggregate display.
