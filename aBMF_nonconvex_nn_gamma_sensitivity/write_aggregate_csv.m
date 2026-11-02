function write_aggregate_csv(aggregate,file_path) % Write the compact method-level rebuttal table to CSV.
file_id = fopen(file_path,'w'); % Open the compact aggregate CSV file.
if file_id < 0 % Check whether the output file was opened successfully.
    error('Cannot create output file: %s',file_path); % Stop when the compact CSV file cannot be created.
end % Finish the file-opening check.
cleanup_object = onCleanup(@() fclose(file_id)); % Guarantee that the file is closed.
fprintf(file_id,'method,num_runs,time_mean,time_std,final_gamma_mean,final_gamma_std,surrogate_objective_mean,surrogate_objective_std,margin_surrogate_mean,margin_surrogate_std,affine_residual_mean,affine_residual_std,inequality_residual_mean,inequality_residual_std,complementarity_residual_mean,complementarity_residual_std,rel_disagreement_mean,rel_disagreement_std,psd_violation_mean,psd_violation_std,success_rate\n'); % Write the compact CSV header.
for row_index = 1:numel(aggregate) % Write every method-level aggregate row.
    row = aggregate(row_index); % Read the current aggregate row.
    fprintf(file_id,'%s,%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n',row.method,row.num_runs,row.time_mean,row.time_std,row.final_gamma_mean,row.final_gamma_std,row.surrogate_objective_mean,row.surrogate_objective_std,row.margin_surrogate_mean,row.margin_surrogate_std,row.affine_residual_mean,row.affine_residual_std,row.inequality_residual_mean,row.inequality_residual_std,row.complementarity_residual_mean,row.complementarity_residual_std,row.rel_disagreement_mean,row.rel_disagreement_std,row.psd_violation_mean,row.psd_violation_std,row.success_rate); % Write all compact aggregate statistics.
end % Finish writing aggregate rows.
clear cleanup_object; % Close the compact CSV file.
end % Finish aggregate CSV writing.
