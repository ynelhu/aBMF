function write_summary_csv(summary,file_path) % Write condition-level means and standard deviations to CSV.
file_id = fopen(file_path,'w'); % Open the summary CSV file.
if file_id < 0 % Check whether the output file was opened successfully.
    error('Cannot create output file: %s',file_path); % Stop when the summary CSV file cannot be created.
end % Finish the file-opening check.
cleanup_object = onCleanup(@() fclose(file_id)); % Guarantee that the file is closed.
fprintf(file_id,'sample_index,epsilon,rank,method,num_trials,iterations_mean,iterations_std,time_mean,time_std,final_gamma_mean,final_gamma_std,final_gamma_hat_mean,final_gamma_hat_std,surrogate_objective_mean,surrogate_objective_std,margin_surrogate_mean,margin_surrogate_std,affine_residual_mean,affine_residual_std,inequality_residual_mean,inequality_residual_std,complementarity_residual_mean,complementarity_residual_std,rel_disagreement_mean,rel_disagreement_std,psd_violation_mean,psd_violation_std,success_rate\n'); % Write the summary CSV header.
for row_index = 1:numel(summary) % Write every condition-level summary row.
    row = summary(row_index); % Read the current summary row.
    fprintf(file_id,'%d,%.16g,%d,%s,%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n',row.sample_index,row.epsilon,row.rank,row.method,row.num_trials,row.iterations_mean,row.iterations_std,row.time_mean,row.time_std,row.final_gamma_mean,row.final_gamma_std,row.final_gamma_hat_mean,row.final_gamma_hat_std,row.surrogate_objective_mean,row.surrogate_objective_std,row.margin_surrogate_mean,row.margin_surrogate_std,row.affine_residual_mean,row.affine_residual_std,row.inequality_residual_mean,row.inequality_residual_std,row.complementarity_residual_mean,row.complementarity_residual_std,row.rel_disagreement_mean,row.rel_disagreement_std,row.psd_violation_mean,row.psd_violation_std,row.success_rate); % Write all condition-level statistics.
end % Finish writing condition-level summary rows.
clear cleanup_object; % Close the summary CSV file.
end % Finish summary CSV writing.
