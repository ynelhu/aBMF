function summary = example_nonconvex_nn() % Run the full nonconvex neural-network verification benchmark.
root_dir = fileparts(mfilename('fullpath')); % Locate the project directory containing this entry point.
summary = run_nonconvex_nn_benchmark('full',root_dir); % Run the full benchmark with the default rebuttal settings.
end % Finish the full benchmark entry point.
