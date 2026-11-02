function aggregate = example_nn_gamma_sensitivity() % Run the full controlled fixed-gamma multiplier sensitivity experiment.
root_dir = fileparts(mfilename('fullpath')); % Locate the project directory containing this entry point.
aggregate = run_nn_gamma_multiplier_sensitivity('full',root_dir); % Run the full multiplier sensitivity experiment.
end % Finish the full sensitivity entry point.
