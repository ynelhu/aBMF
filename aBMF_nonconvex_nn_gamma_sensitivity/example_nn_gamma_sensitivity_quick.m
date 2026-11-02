function aggregate = example_nn_gamma_sensitivity_quick() % Run a short smoke test of the fixed-gamma multiplier sensitivity experiment.
root_dir = fileparts(mfilename('fullpath')); % Locate the project directory containing this entry point.
aggregate = run_nn_gamma_multiplier_sensitivity('quick',root_dir); % Run the reduced multiplier sensitivity experiment.
end % Finish the quick sensitivity entry point.
