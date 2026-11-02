function write_gamma_sensitivity_matched_csv(rows,file_name) % Write matched changes relative to c=1 to CSV.
field_names = fieldnames(rows); % Read the stable matched-summary field order.
write_numeric_structure(rows,field_names,file_name); % Write the complete matched summary.
end % Finish the matched-summary CSV writer.

function write_numeric_structure(rows,field_names,file_name) % Write a numeric structure array with a header row.
file_id = fopen(file_name,'w'); % Open the matched-summary CSV file for writing.
if file_id < 0 % Check whether the output file opened successfully.
    error('Cannot open output file: %s',file_name); % Stop when the matched-summary file cannot be opened.
end % Finish the file-open check.
cleanup_object = onCleanup(@() fclose(file_id)); % Ensure that the file is closed when this function exits.
for field_index = 1:numel(field_names) % Write every header field.
    if field_index > 1 % Check whether a separator is needed.
        fprintf(file_id,','); % Write a comma separator.
    end % Finish the separator check.
    fprintf(file_id,'%s',field_names{field_index}); % Write the current header field.
end % Finish writing the header fields.
fprintf(file_id,'\n'); % Finish the header row.
for row_index = 1:numel(rows) % Write every matched-summary row.
    for field_index = 1:numel(field_names) % Write every numeric matched-summary field.
        if field_index > 1 % Check whether a separator is needed.
            fprintf(file_id,','); % Write a comma separator.
        end % Finish the separator check.
        fprintf(file_id,'%.17g',rows(row_index).(field_names{field_index})); % Write the current numeric matched-summary value.
    end % Finish writing the current matched-summary fields.
    fprintf(file_id,'\n'); % Finish the current matched-summary row.
end % Finish writing all matched-summary rows.
clear cleanup_object; % Close the matched-summary file explicitly.
end % Finish the generic numeric-structure writer.
