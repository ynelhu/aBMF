function dataset = load_digits_network(data_file) % Load the bundled Digits 3-versus-8 data and fixed ReLU network.
if ~exist(data_file,'file') % Check whether the prepared data file exists.
    error('Missing prepared data file: %s',data_file); % Stop when the bundled data file is absent.
end % Finish the data-file check.
loaded = load(data_file); % Load all prepared arrays and network parameters.
required_fields = {'X_train','y_train','X_test','y_test','W1','b1','w2','b2','selected_test_indices','test_accuracy'}; % Define the required MAT-file variables.
for field_index = 1:numel(required_fields) % Validate every required variable.
    if ~isfield(loaded,required_fields{field_index}) % Check whether the current variable exists.
        error('The prepared data file is missing variable %s.',required_fields{field_index}); % Stop for an incomplete data file.
    end % Finish the variable-existence check.
end % Finish validating the MAT file.
dataset.X_train = double(loaded.X_train); % Store the transformed training inputs as columns.
dataset.y_train = double(loaded.y_train(:)); % Store the binary training labels as a column.
dataset.X_test = double(loaded.X_test); % Store the transformed test inputs as columns.
dataset.y_test = double(loaded.y_test(:)); % Store the binary test labels as a column.
dataset.W1 = double(loaded.W1); % Store the first-layer weight matrix.
dataset.b1 = double(loaded.b1(:)); % Store the first-layer bias vector.
dataset.w2 = double(loaded.w2(:)); % Store the output-layer weight vector.
dataset.b2 = double(loaded.b2(1)); % Store the scalar output bias.
dataset.selected_test_indices = round(double(loaded.selected_test_indices(:))); % Store the three selected test-set indices.
dataset.test_accuracy = double(loaded.test_accuracy(1)); % Store the fixed test accuracy.
dataset.input_dimension = size(dataset.W1,2); % Record the transformed input dimension.
dataset.hidden_units = size(dataset.W1,1); % Record the number of ReLU hidden units.
end % Finish the prepared-data loader.
