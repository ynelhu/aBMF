function write_raw_csv(raw_results,file_path) % Write all individual benchmark runs to a CSV file.
file_id = fopen(file_path,'w'); % Open the raw-result CSV file.
if file_id < 0 % Check whether the output file was opened successfully.
    error('Cannot create output file: %s',file_path); % Stop when the raw CSV file cannot be created.
end % Finish the file-opening check.
cleanup_object = onCleanup(@() fclose(file_id)); % Guarantee that the file is closed.
fprintf(file_id,'sample_index,epsilon,rank,method,trial,iterations,time,final_gamma,final_gamma_hat,surrogate_objective,margin_surrogate,affine_residual,inequality_residual,complementarity_residual,rel_disagreement,psd_violation,success\n'); % Write the raw CSV header.
for row_index = 1:numel(raw_results) % Write every raw benchmark row.
    row = raw_results(row_index); % Read the current raw result.
    fprintf(file_id,'%d,%.16g,%d,%s,%d,%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%d\n',row.sample_index,row.epsilon,row.rank,row.method,row.trial,row.iterations,row.time,row.final_gamma,row.final_gamma_hat,row.surrogate_objective,row.margin_surrogate,row.affine_residual,row.inequality_residual,row.complementarity_residual,row.rel_disagreement,row.psd_violation,row.success); % Write the complete raw result.
end % Finish writing raw benchmark rows.
clear cleanup_object; % Close the raw-result CSV file.
end % Finish raw CSV writing.
