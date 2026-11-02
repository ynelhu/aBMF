function plot_representative_histories(history_records,output_dir) % Plot representative penalty and residual histories.
figure_handle = figure('Visible','off'); % Create an off-screen figure for the gamma trajectories.
hold on; % Retain all asymmetric gamma trajectories.
legend_entries = cell(0,1); % Allocate the gamma-plot legend entries.
for record_index = 1:numel(history_records) % Process every stored representative history.
    record = history_records(record_index); % Read the current history record.
    if isempty(record.history) || all(isnan(record.history.gamma)) % Check whether the method has a meaningful gamma trajectory.
        continue; % Skip symmetric methods in the gamma figure.
    end % Finish the gamma-history check.
    semilogy(record.history.iteration,max(record.history.gamma,1e-16),'LineWidth',1.2); % Plot the positive gamma trajectory on a logarithmic scale.
    legend_entries{end+1,1} = record.method; % Store the corresponding legend entry.
end % Finish plotting all gamma histories.
xlabel('Iteration'); % Label the horizontal axis.
ylabel('\gamma_k'); % Label the vertical axis.
title('Representative consistency-penalty trajectories'); % Add a descriptive figure title.
grid on; % Display a reference grid.
if ~isempty(legend_entries) % Check whether any asymmetric trajectory was plotted.
    legend(legend_entries,'Location','best'); % Display the method legend.
end % Finish the legend check.
saveas(figure_handle,fullfile(output_dir,'gamma_trajectory.png')); % Save the gamma-trajectory figure.
close(figure_handle); % Close the off-screen gamma figure.
figure_handle = figure('Visible','off'); % Create an off-screen figure for complementarity residuals.
hold on; % Retain all complementarity trajectories.
legend_entries = cell(0,1); % Reset the legend entries.
for record_index = 1:numel(history_records) % Process every stored representative history.
    record = history_records(record_index); % Read the current history record.
    if isempty(record.history) % Check whether the history is empty.
        continue; % Skip empty history records.
    end % Finish the empty-history check.
    semilogy(record.history.iteration,max(record.history.complementarity,1e-16),'LineWidth',1.2); % Plot the complementarity residual trajectory.
    legend_entries{end+1,1} = record.method; % Store the corresponding legend entry.
end % Finish plotting all complementarity histories.
xlabel('Iteration'); % Label the horizontal axis.
ylabel('Complementarity residual'); % Label the vertical axis.
title('Representative nonconvex ReLU complementarity residuals'); % Add a descriptive figure title.
grid on; % Display a reference grid.
if ~isempty(legend_entries) % Check whether any trajectory was plotted.
    legend(legend_entries,'Location','best'); % Display the method legend.
end % Finish the legend check.
saveas(figure_handle,fullfile(output_dir,'complementarity_trajectory.png')); % Save the complementarity-residual figure.
close(figure_handle); % Close the off-screen complementarity figure.
figure_handle = figure('Visible','off'); % Create an off-screen figure for factor disagreements.
hold on; % Retain all disagreement trajectories.
legend_entries = cell(0,1); % Reset the legend entries.
for record_index = 1:numel(history_records) % Process every stored representative history.
    record = history_records(record_index); % Read the current history record.
    if isempty(record.history) % Check whether the history is empty.
        continue; % Skip empty history records.
    end % Finish the empty-history check.
    semilogy(record.history.iteration,max(record.history.disagreement,1e-16),'LineWidth',1.2); % Plot the relative factor disagreement.
    legend_entries{end+1,1} = record.method; % Store the corresponding legend entry.
end % Finish plotting all disagreement histories.
xlabel('Iteration'); % Label the horizontal axis.
ylabel('Relative factor disagreement'); % Label the vertical axis.
title('Representative asymmetric factor-disagreement trajectories'); % Add a descriptive figure title.
grid on; % Display a reference grid.
if ~isempty(legend_entries) % Check whether any trajectory was plotted.
    legend(legend_entries,'Location','best'); % Display the method legend.
end % Finish the legend check.
saveas(figure_handle,fullfile(output_dir,'factor_disagreement_trajectory.png')); % Save the factor-disagreement figure.
close(figure_handle); % Close the off-screen disagreement figure.
end % Finish representative history plotting.
