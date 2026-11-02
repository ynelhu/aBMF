function write_gamma_sensitivity_aggregate_csv(rows,file_name) % Write multiplier-level gamma-sensitivity aggregates to CSV.
field_names = fieldnames(rows); % Read the stable multiplier-level field order.
write_numeric_structure(rows,field_names,file_name); % Write the complete multiplier-level aggregate.
end % Finish the multiplier-level aggregate CSV writer.

function write_numeric_structure(rows,field_names,file_name) % Write a numeric structure array with a header row.
file_id = fopen(file_name,'w'); % Open the multiplier-level CSV file for writing.
if file_id < 0 % Check whether the output file opened successfully.
    error('Cannot open output file: %s',file_name); % Stop when the multiplier-level file cannot be opened.
end % Finish the file-open check.
cleanup_object = onCleanup(@() fclose(file_id)); % Ensure that the file is closed when this function exits.
for field_index = 1:numel(field_names) % Write every header field.
    if field_index > 1 % Check whether a separator is needed.
        fprintf(file_id,','); % Write a comma separator.
    end % Finish the separator check.
    fprintf(file_id,'%s',field_names{field_index}); % Write the current header field.
end % Finish writing the header fields.
fprintf(file_id,'\n'); % Finish the header row.
for row_index = 1:numel(rows) % Write every aggregate row.
    for field_index = 1:numel(field_names) % Write every numeric aggregate field.
        if field_index > 1 % Check whether a separator is needed.
            fprintf(file_id,','); % Write a comma separator.
        end % Finish the separator check.
        fprintf(file_id,'%.17g',rows(row_index).(field_names{field_index})); % Write the current numeric aggregate value.
    end % Finish writing the current aggregate fields.
    fprintf(file_id,'\n'); % Finish the current aggregate row.
end % Finish writing all aggregate rows.
clear cleanup_object; % Close the multiplier-level file explicitly.
end % Finish the generic numeric-structure writer.
