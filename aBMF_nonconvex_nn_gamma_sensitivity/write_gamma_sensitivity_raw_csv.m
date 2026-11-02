function write_gamma_sensitivity_raw_csv(rows,file_name) % Write all individual gamma-sensitivity runs to a CSV file.
file_id = fopen(file_name,'w'); % Open the raw-result CSV file for writing.
if file_id < 0 % Check whether the output file opened successfully.
    error('Cannot open output file: %s',file_name); % Stop when the raw-result file cannot be opened.
end % Finish the file-open check.
cleanup_object = onCleanup(@() fclose(file_id)); % Ensure that the file is closed when this function exits.
fprintf(file_id,'sample_index,epsilon,rank,trial,gamma_multiplier,gamma_reference,gamma_value,iterations,time,surrogate_objective,margin_surrogate,affine_residual,inequality_residual,complementarity_residual,rel_disagreement,psd_violation,success\n'); % Write the raw-result header.
for row_index = 1:numel(rows) % Write every individual controlled run.
    row = rows(row_index); % Read the current raw-result row.
    fprintf(file_id,'%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n',row.sample_index,row.epsilon,row.rank,row.trial,row.gamma_multiplier,row.gamma_reference,row.gamma_value,row.iterations,row.time,row.surrogate_objective,row.margin_surrogate,row.affine_residual,row.inequality_residual,row.complementarity_residual,row.rel_disagreement,row.psd_violation,row.success); % Write the current raw-result values.
end % Finish writing all raw-result rows.
clear cleanup_object; % Close the raw-result file explicitly.
end % Finish the raw-result CSV writer.
