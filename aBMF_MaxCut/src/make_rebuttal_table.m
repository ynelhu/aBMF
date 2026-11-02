function compact = make_rebuttal_table(summary_source,output_file) % Convert the full Max-Cut summary into a compact rebuttal table.
if nargin<1 || isempty(summary_source) % Check whether a summary source was supplied.
    summary_source = fullfile(pwd,'maxcut_results','summary_mean_std.csv'); % Use the default result path.
end % Finish the summary-source branch.
if nargin<2 || isempty(output_file) % Check whether an output file was supplied.
    output_file = fullfile(fileparts(summary_source),'rebuttal_table.csv'); % Use a neighboring compact-table path.
end % Finish the output-file branch.
if ischar(summary_source) % Check whether the source is a CSV path.
    summary = readtable(summary_source); % Read the full statistical summary.
else % Handle an already loaded MATLAB table.
    summary = summary_source; % Copy the supplied table.
end % Finish the summary-source type branch.
row_count = height(summary); % Read the number of summary rows.
time_text = cell(row_count,1); % Preallocate formatted runtime values.
objective_text = cell(row_count,1); % Preallocate formatted feasible SDP objective values.
bilinear_text = cell(row_count,1); % Preallocate formatted asymmetric bilinear objective values.
cut_ratio_text = cell(row_count,1); % Preallocate formatted cut-ratio values.
disagreement_text = cell(row_count,1); % Preallocate formatted disagreement values.
feasibility_text = cell(row_count,1); % Preallocate formatted diagonal-equality residual values.
gamma_text = cell(row_count,1); % Preallocate formatted final gamma values.
candidate_text = cell(row_count,1); % Preallocate formatted gradient-norm gamma candidates.
for row_index = 1:row_count % Loop over summary rows.
    time_text{row_index} = format_mean_std(summary.time_mean(row_index),summary.time_std(row_index)); % Format runtime as mean plus or minus standard deviation.
    objective_text{row_index} = format_mean_std(summary.sdp_objective_mean(row_index),summary.sdp_objective_std(row_index)); % Format the feasible SDP objective.
    bilinear_text{row_index} = format_mean_std(summary.bilinear_sdp_objective_mean(row_index),summary.bilinear_sdp_objective_std(row_index)); % Format the asymmetric bilinear objective.
    cut_ratio_text{row_index} = format_mean_std(summary.cut_ratio_mean(row_index),summary.cut_ratio_std(row_index)); % Format the cut ratio.
    disagreement_text{row_index} = format_mean_std(summary.rel_disagreement_mean(row_index),summary.rel_disagreement_std(row_index)); % Format relative disagreement.
    feasibility_text{row_index} = format_mean_std(summary.equality_residual_mean(row_index),summary.equality_residual_std(row_index)); % Format diagonal-equality feasibility.
    gamma_text{row_index} = format_mean_std(summary.final_gamma_mean(row_index),summary.final_gamma_std(row_index)); % Format the final agreement penalty.
    candidate_text{row_index} = format_mean_std(summary.gamma_candidate_mean(row_index),summary.gamma_candidate_std(row_index)); % Format the final gradient-norm gamma candidate.
end % Finish the row-formatting loop.
compact = table(summary.graph_name,summary.n,summary.rank,summary.method,time_text,objective_text,bilinear_text,cut_ratio_text,disagreement_text,feasibility_text,gamma_text,candidate_text,'VariableNames',{'graph_name','n','rank','method','time','sdp_objective','bilinear_objective','cut_ratio','rel_disagreement','equality_residual','final_gamma','gamma_candidate'}); % Build the compact rebuttal table.
writetable(compact,output_file); % Save the compact rebuttal table.
disp(compact); % Display the compact table.
end % Finish the compact rebuttal-table function.

function text = format_mean_std(average,deviation) % Format one mean-and-standard-deviation pair.
if ~isfinite(average) % Check whether the metric is unavailable.
    text = '--'; % Use a compact unavailable marker.
else % Handle an available metric.
    text = sprintf('%.4g +/- %.2g',average,deviation); % Format the mean and standard deviation compactly.
end % Finish the metric-availability branch.
end % Finish the mean-and-standard-deviation formatter.
