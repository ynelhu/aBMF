function plot_gamma_sensitivity_aggregate(aggregate,output_dir) % Generate four separate multiplier-sensitivity figures.
multipliers = extract_aggregate_field(aggregate,'gamma_multiplier'); % Extract the controlled multiplier values.
plot_one_metric(multipliers,extract_aggregate_field(aggregate,'surrogate_objective_mean'),extract_aggregate_field(aggregate,'surrogate_objective_std'),'Fixed-gamma multiplier c','Surrogate objective','gamma_sensitivity_objective.png',output_dir); % Plot the surrogate-objective sensitivity.
plot_one_metric(multipliers,extract_aggregate_field(aggregate,'rel_disagreement_mean'),extract_aggregate_field(aggregate,'rel_disagreement_std'),'Fixed-gamma multiplier c','Relative factor disagreement','gamma_sensitivity_disagreement.png',output_dir); % Plot the factor-disagreement sensitivity.
plot_one_metric(multipliers,extract_aggregate_field(aggregate,'iterations_mean'),extract_aggregate_field(aggregate,'iterations_std'),'Fixed-gamma multiplier c','Iterations','gamma_sensitivity_iterations.png',output_dir); % Plot the iteration-count sensitivity.
plot_one_metric(multipliers,extract_aggregate_field(aggregate,'time_mean'),extract_aggregate_field(aggregate,'time_std'),'Fixed-gamma multiplier c','Runtime (seconds)','gamma_sensitivity_runtime.png',output_dir); % Plot the runtime sensitivity.
end % Finish the aggregate sensitivity plotting routine.

function plot_one_metric(x_values,y_values,error_values,x_label,y_label,file_name,output_dir) % Generate one semilogarithmic sensitivity figure.
figure_handle = figure('Visible','off'); % Create an invisible figure for noninteractive execution.
errorbar(x_values,y_values,error_values,'-o','LineWidth',1.2,'MarkerSize',6); % Plot means with one-standard-deviation error bars.
set(gca,'XScale','log'); % Use a logarithmic multiplier axis.
set(gca,'XTick',x_values); % Show every tested multiplier explicitly.
xlabel(x_label); % Label the horizontal axis.
ylabel(y_label); % Label the vertical axis.
grid on; % Add a reference grid.
box on; % Draw the plot border.
print(figure_handle,fullfile(output_dir,file_name),'-dpng','-r200'); % Save the current sensitivity figure as a PNG file.
close(figure_handle); % Close the current figure.
end % Finish the single-metric plotting helper.

function values = extract_aggregate_field(aggregate,field_name) % Extract one aggregate field as a column vector.
values = zeros(numel(aggregate),1); % Allocate the numeric output vector.
for row_index = 1:numel(aggregate) % Read every aggregate row.
    values(row_index) = aggregate(row_index).(field_name); % Store the requested aggregate field.
end % Finish extracting the aggregate field.
end % Finish the aggregate-field extractor.
