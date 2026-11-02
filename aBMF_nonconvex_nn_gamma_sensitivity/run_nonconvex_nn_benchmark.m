function summary = run_nonconvex_nn_benchmark(run_mode,root_dir) % Run the complete nonconvex PSD-lifted neural-network experiment.
if nargin < 1 || isempty(run_mode) % Check whether a run mode was supplied.
    run_mode = 'full'; % Use the full benchmark by default.
end % Finish the run-mode check.
if nargin < 2 || isempty(root_dir) % Check whether a project directory was supplied.
    root_dir = fileparts(mfilename('fullpath')); % Use the directory containing this function by default.
end % Finish the project-directory check.
settings = default_settings(run_mode,root_dir); % Construct all experimental settings in one structure.
dataset = load_digits_network(settings.data_file); % Load the prepared Digits 3-versus-8 network and data.
fprintf('Prepared Digits 3-vs-8 network test accuracy: %.4f.\n',dataset.test_accuracy); % Display the fixed classifier accuracy.
if ~exist(settings.output_dir,'dir') % Check whether the result directory exists.
    mkdir(settings.output_dir); % Create the result directory when necessary.
end % Finish the result-directory check.
raw_results = repmat(result_row_template(),0,1); % Allocate a fixed-layout result structure.
history_records = repmat(history_record_template(),0,1); % Allocate storage for representative optimization histories.
row_counter = 0; % Initialize the completed-run counter.
history_counter = 0; % Initialize the stored-history counter.
for sample_position = 1:numel(settings.sample_positions) % Process every selected correctly classified test point.
    sample_index = dataset.selected_test_indices(settings.sample_positions(sample_position)); % Read the test-set index of the current sample.
    x0 = dataset.X_test(:,sample_index); % Read the nominal transformed input vector.
    y0 = dataset.y_test(sample_index); % Read the binary class label in plus-or-minus-one form.
    nominal_score = forward_network(dataset,x0); % Evaluate the nominal signed network score.
    nominal_margin = y0*nominal_score; % Compute the nominal signed classification margin.
    for epsilon_index = 1:numel(settings.epsilon_list) % Process every perturbation radius.
        epsilon_value = settings.epsilon_list(epsilon_index); % Read the current infinity-norm perturbation radius.
        problem = build_verification_problem(dataset,x0,y0,epsilon_value,settings); % Construct the nonconvex PSD-lifted surrogate.
        fprintf('\nSample %d, epsilon %.3f, nominal margin %.6g, lift p = %d.\n',sample_index,epsilon_value,nominal_margin,problem.p); % Display the current verification condition.
        for rank_index = 1:numel(settings.rank_list) % Process every requested factor rank.
            rank_value = settings.rank_list(rank_index); % Read the current low-rank factor dimension.
            for trial_index = 1:settings.num_trials % Repeat each randomized low-rank method.
                common_seed = settings.seed_base + 100000*sample_position + 10000*epsilon_index + 100*rank_index + trial_index; % Construct a reproducible common initialization seed.
                for method_index = 1:numel(settings.methods) % Run every compared factorization method.
                    method_name = settings.methods{method_index}; % Read the current method name.
                    fprintf('  r=%d, trial=%d, method=%s ... ',rank_value,trial_index,method_name); % Display the current run.
                    timer_value = tic; % Start measuring wall-clock time.
                    if strcmp(method_name,'sBMF') % Check whether the symmetric factorization was requested.
                        solution = solve_sbmf_nc(problem,rank_value,common_seed,settings); % Run the symmetric nonconvex low-rank solver.
                    else % Handle an asymmetric factorization.
                        solution = solve_abmf_nc(problem,rank_value,common_seed,method_name,settings); % Run the selected asymmetric solver.
                    end % Finish the solver selection.
                    elapsed_time = toc(timer_value); % Obtain the wall-clock time.
                    metrics = evaluate_nc_solution(problem,solution,settings); % Evaluate all methods with one common metric routine.
                    row_counter = row_counter + 1; % Increase the completed-run counter.
                    raw_results(row_counter,1) = result_row_template(); % Create a result row with a fixed field layout.
                    raw_results(row_counter).sample_index = sample_index; % Store the selected test-set index.
                    raw_results(row_counter).epsilon = epsilon_value; % Store the perturbation radius.
                    raw_results(row_counter).rank = rank_value; % Store the factor rank.
                    raw_results(row_counter).method = method_name; % Store the method name.
                    raw_results(row_counter).trial = trial_index; % Store the trial index.
                    raw_results(row_counter).iterations = solution.iterations; % Store the completed iteration number.
                    raw_results(row_counter).time = elapsed_time; % Store the wall-clock time.
                    raw_results(row_counter).final_gamma = solution.final_gamma; % Store the final consistency penalty.
                    raw_results(row_counter).final_gamma_hat = solution.final_gamma_hat; % Store the final strict gradient-based penalty candidate.
                    raw_results(row_counter).surrogate_objective = metrics.surrogate_objective; % Store the complete nonconvex surrogate objective.
                    raw_results(row_counter).margin_surrogate = metrics.margin_surrogate; % Store the signed verification-margin surrogate.
                    raw_results(row_counter).affine_residual = metrics.affine_residual; % Store the maximum affine residual.
                    raw_results(row_counter).inequality_residual = metrics.inequality_residual; % Store the maximum inequality violation.
                    raw_results(row_counter).complementarity_residual = metrics.complementarity_residual; % Store the ReLU complementarity residual.
                    raw_results(row_counter).rel_disagreement = metrics.rel_disagreement; % Store the relative factor disagreement.
                    raw_results(row_counter).psd_violation = metrics.psd_violation; % Store the negative-eigenvalue violation of the symmetrized split.
                    raw_results(row_counter).success = metrics.success; % Store the common diagnostic success flag.
                    fprintf('obj=%.5g, comp=%.3e, gap=%.3e, time=%.3fs.\n',metrics.surrogate_objective,metrics.complementarity_residual,metrics.rel_disagreement,elapsed_time); % Display the main diagnostics.
                    if settings.save_histories && trial_index == 1 && rank_index == 1 && sample_position == settings.history_sample_position && epsilon_index == settings.history_epsilon_position % Check whether this run is the designated representative history.
                        history_counter = history_counter + 1; % Increase the stored-history counter.
                        history_records(history_counter,1) = history_record_template(); % Create a fixed-layout history record.
                        history_records(history_counter).method = method_name; % Store the method name for the history.
                        history_records(history_counter).sample_index = sample_index; % Store the representative sample index.
                        history_records(history_counter).epsilon = epsilon_value; % Store the representative perturbation radius.
                        history_records(history_counter).rank = rank_value; % Store the representative factor rank.
                        history_records(history_counter).history = solution.history; % Store the complete optimization history.
                    end % Finish the representative-history check.
                end % Finish the method loop.
            end % Finish the trial loop.
        end % Finish the rank loop.
    end % Finish the perturbation-radius loop.
end % Finish the selected-sample loop.
summary = summarize_nc_results(raw_results); % Aggregate means and standard deviations by experimental condition.
write_raw_csv(raw_results,fullfile(settings.output_dir,'raw_results.csv')); % Write all individual runs to a CSV file.
write_summary_csv(summary,fullfile(settings.output_dir,'summary_mean_std.csv')); % Write the condition-level mean and standard deviation table.
aggregate = aggregate_rebuttal_results(raw_results); % Aggregate all selected conditions into one compact method-level table.
write_aggregate_csv(aggregate,fullfile(settings.output_dir,'rebuttal_aggregate.csv')); % Write the compact rebuttal-oriented table.
save(fullfile(settings.output_dir,'results_and_histories.mat'),'raw_results','summary','aggregate','history_records','settings'); % Save complete MATLAB result structures.
if settings.save_histories && ~isempty(history_records) % Check whether representative histories were stored.
    plot_representative_histories(history_records,settings.output_dir); % Generate diagnostic history figures.
end % Finish the history-plot check.
fprintf('\nExperiments completed. Results were saved to:\n%s\n',settings.output_dir); % Report successful completion and the output directory.
display_aggregate(aggregate); % Display the compact method-level summary in the command window.
end % Finish the benchmark driver.

function settings = default_settings(run_mode,root_dir) % Construct all default settings for the rebuttal experiment.
settings.data_file = fullfile(root_dir,'data','digits_3_8_prepared.mat'); % Specify the bundled prepared Digits dataset and network.
settings.output_dir = fullfile(root_dir,'results'); % Specify the output directory.
settings.methods = {'sBMF','aBMF-fixed','aBMF-adp','aBMF-our'}; % Define the compared symmetric and asymmetric methods.
settings.sample_positions = 1:3; % Use the three margin-quantile test samples in the full benchmark.
settings.epsilon_list = [0.6,0.8]; % Use two infinity-norm perturbation radii.
settings.rank_list = [8,16]; % Use two factor ranks.
settings.num_trials = 10; % Repeat every randomized low-rank condition ten times for the rebuttal table.
settings.max_iter = 3000; % Allow up to three thousand outer iterations in the full benchmark.
settings.tol = 1e-6; % Set the relative factor-change stopping tolerance.
settings.initial_step = 1e-3; % Set the initial gradient step for every block.
settings.maximum_step = 1e-2; % Cap the accepted gradient step.
settings.armijo = 1e-4; % Set the sufficient-decrease coefficient.
settings.backtracking_factor = 0.5; % Set the geometric backtracking reduction.
settings.maximum_backtracking = 25; % Limit the number of line-search reductions.
settings.factor_norm_cap = 200; % Cap the factor Frobenius norm for numerical protection.
settings.rho_eq = 50; % Weight the affine-equality penalty.
settings.rho_ineq = 50; % Weight the one-sided inequality penalty.
settings.rho_comp = 100; % Weight the nonconvex ReLU complementarity penalty.
settings.lambda_fro = 1e-3; % Weight the lifted-matrix Frobenius regularizer.
settings.gamma0 = 10; % Use one common initial consistency penalty for all asymmetric methods.
settings.gamma_epsilon = 1e-6; % Add the strict epsilon safeguard to the gradient-based gamma candidate.
settings.gamma_max = 1e4; % Cap the multiplicative adaptive baseline.
settings.epsilon_tau = 1e-12; % Protect the denominator of the multiplicative adaptive baseline.
settings.seed_base = 314159; % Define the common random-seed base.
settings.success_tolerance = 5e-2; % Define one common diagnostic residual threshold.
settings.save_histories = true; % Save representative optimization histories.
settings.history_sample_position = 2; % Use the median-margin sample for history plots.
settings.history_epsilon_position = 2; % Use epsilon equal to the second listed radius for history plots.
if strcmpi(run_mode,'quick') % Check whether the user requested a smoke test.
    settings.sample_positions = 2; % Use only the median-margin sample.
    settings.epsilon_list = 0.8; % Use one perturbation radius.
    settings.rank_list = 8; % Use one factor rank.
    settings.num_trials = 1; % Use one randomized trial.
    settings.max_iter = 80; % Use a short iteration limit.
    settings.history_sample_position = 1; % Store the only sample history.
    settings.history_epsilon_position = 1; % Store the only epsilon history.
elseif ~strcmpi(run_mode,'full') % Reject unsupported run-mode names.
    error('Unknown run mode: %s. Use ''full'' or ''quick''.',run_mode); % Stop for an unsupported run mode.
end % Finish the run-mode selection.
end % Finish the default-setting constructor.

function row = result_row_template() % Create one fixed-layout raw-result structure.
row = struct('sample_index',0,'epsilon',NaN,'rank',0,'method','','trial',0,'iterations',0,'time',NaN,'final_gamma',NaN,'final_gamma_hat',NaN,'surrogate_objective',NaN,'margin_surrogate',NaN,'affine_residual',NaN,'inequality_residual',NaN,'complementarity_residual',NaN,'rel_disagreement',NaN,'psd_violation',NaN,'success',0); % Define all raw-result fields in a stable order.
end % Finish the raw-result template.

function record = history_record_template() % Create one fixed-layout optimization-history record.
record = struct('method','','sample_index',0,'epsilon',NaN,'rank',0,'history',[]); % Define all history-record fields in a stable order.
end % Finish the history-record template.
