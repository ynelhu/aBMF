function self_check_gamma_sensitivity() % Perform static file and configuration checks for the multiplier experiment.
root_dir = fileparts(mfilename('fullpath')); % Locate the project directory containing this check.
required_files = {'run_nn_gamma_multiplier_sensitivity.m','summarize_gamma_sensitivity_by_condition.m','summarize_gamma_sensitivity_overall.m','summarize_gamma_sensitivity_matched.m','write_gamma_sensitivity_raw_csv.m','write_gamma_sensitivity_condition_csv.m','write_gamma_sensitivity_aggregate_csv.m','write_gamma_sensitivity_matched_csv.m','plot_gamma_sensitivity_aggregate.m','display_gamma_sensitivity_aggregate.m','solve_abmf_nc.m','evaluate_nonconvex_lift.m','evaluate_nc_solution.m','build_verification_problem.m','load_digits_network.m','data/digits_3_8_prepared.mat'}; % Define all required sensitivity files.
for file_index = 1:numel(required_files) % Check every required file.
    file_path = fullfile(root_dir,required_files{file_index}); % Construct the current required path.
    if exist(file_path,'file') ~= 2 % Check whether the required file exists.
        error('Missing required file: %s',file_path); % Stop when a required file is absent.
    end % Finish the required-file check.
end % Finish checking all required files.
fprintf('Static gamma-sensitivity file checks passed.\n'); % Report successful static checks.
fprintf('Run example_nn_gamma_sensitivity_quick before the full experiment.\n'); % Display the recommended next command.
end % Finish the sensitivity self-check.
