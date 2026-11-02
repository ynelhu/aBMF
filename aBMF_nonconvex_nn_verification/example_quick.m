function summary = example_quick() % Run a short smoke test of the nonconvex neural-network benchmark.
root_dir = fileparts(mfilename('fullpath')); % Locate the project directory containing this entry point.
summary = run_nonconvex_nn_benchmark('quick',root_dir); % Run a reduced benchmark for installation and syntax checking.
end % Finish the quick benchmark entry point.
