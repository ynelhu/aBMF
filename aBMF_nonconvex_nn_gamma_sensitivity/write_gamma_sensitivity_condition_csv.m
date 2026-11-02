function write_gamma_sensitivity_condition_csv(rows,file_name) % Write condition-level gamma-sensitivity means and standard deviations to CSV.
field_names = fieldnames(rows); % Read the stable condition-summary field order.
write_structure_csv(rows,field_names,file_name); % Write the complete condition-summary structure.
end % Finish the condition-summary CSV writer.

function write_structure_csv(rows,field_names,file_name) % Write a numeric structure array to a CSV file.
file_id = fopen(file_name,'w'); % Open the condition-summary CSV file for writing.
if file_id < 0 % Check whether the output file opened successfully.
    error('Cannot open output file: %s',file_name); % Stop when the condition-summary file cannot be opened.
end % Finish the file-open check.
cleanup_object = onCleanup(@() fclose(file_id)); % Ensure that the file is closed when this function exits.
for field_index = 1:numel(field_names) % Write every header field.
    if field_index > 1 % Check whether a separator is needed.
        fprintf(file_id,','); % Write a comma separator.
    end % Finish the separator check.
    fprintf(file_id,'%s',field_names{field_index}); % Write the current header field.
end % Finish writing the header fields.
fprintf(file_id,'\n'); % Finish the header row.
for row_index = 1:numel(rows) % Write every summary row.
    for field_index = 1:numel(field_names) % Write every numeric summary field.
        if field_index > 1 % Check whether a separator is needed.
            fprintf(file_id,','); % Write a comma separator.
        end % Finish the separator check.
        value = rows(row_index).(field_names{field_index}); % Read the current numeric value.
        fprintf(file_id,'%.17g',value); % Write the current numeric value.
    end % Finish writing the current summary fields.
    fprintf(file_id,'\n'); % Finish the current summary row.
end % Finish writing all summary rows.
clear cleanup_object; % Close the condition-summary file explicitly.
end % Finish the generic numeric-structure CSV writer.
