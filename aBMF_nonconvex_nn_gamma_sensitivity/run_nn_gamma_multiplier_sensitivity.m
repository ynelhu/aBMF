function aggregate = run_nn_gamma_multiplier_sensitivity(run_mode,root_dir) % Run a controlled fixed-gamma multiplier sensitivity experiment on the nonconvex NN surrogate.
if nargin < 1 || isempty(run_mode) % Check whether the caller supplied a run mode.
    run_mode = 'full'; % Use the full sensitivity experiment by default.
end % Finish the run-mode check.
if nargin < 2 || isempty(root_dir) % Check whether the caller supplied the project directory.
    root_dir = fileparts(mfilename('fullpath')); % Use the directory containing this function by default.
end % Finish the project-directory check.
settings = sensitivity_default_settings(run_mode,root_dir); % Construct all controlled experimental settings.
dataset = load_digits_network(settings.data_file); % Load the bundled Digits 3-versus-8 network and data.
fprintf('Prepared Digits 3-vs-8 network test accuracy: %.4f.\n',dataset.test_accuracy); % Display the fixed-network test accuracy.
if ~exist(settings.output_dir,'dir') % Check whether the sensitivity-output directory exists.
    mkdir(settings.output_dir); % Create the sensitivity-output directory when necessary.
end % Finish the output-directory check.
raw_results = repmat(sensitivity_result_template(),0,1); % Allocate a fixed-layout raw-result structure.
row_counter = 0; % Initialize the completed-run counter.
for sample_position = 1:numel(settings.sample_positions) % Process every selected correctly classified test point.
    sample_index = dataset.selected_test_indices(settings.sample_positions(sample_position)); % Read the current selected test-set index.
    x0 = dataset.X_test(:,sample_index); % Read the nominal transformed input vector.
    y0 = dataset.y_test(sample_index); % Read the binary class label in plus-or-minus-one form.
    nominal_score = forward_network(dataset,x0); % Evaluate the nominal signed network score.
    nominal_margin = y0*nominal_score; % Compute the nominal signed classification margin.
    for epsilon_index = 1:numel(settings.epsilon_list) % Process every requested perturbation radius.
        epsilon_value = settings.epsilon_list(epsilon_index); % Read the current infinity-norm perturbation radius.
        problem = build_verification_problem(dataset,x0,y0,epsilon_value,settings); % Construct the nonconvex PSD-lifted verification surrogate.
        fprintf('\nSample %d, epsilon %.3f, nominal margin %.6g, lift p = %d.\n',sample_index,epsilon_value,nominal_margin,problem.p); % Display the current verification condition.
        for rank_index = 1:numel(settings.rank_list) % Process every requested factor rank.
            rank_value = settings.rank_list(rank_index); % Read the current low-rank factor dimension.
            for trial_index = 1:settings.num_trials % Repeat every condition with matched randomized initialization.
                common_seed = settings.seed_base + 100000*sample_position + 10000*epsilon_index + 100*rank_index + trial_index; % Construct the reproducible matched initialization seed.
                gamma_reference = compute_initial_gamma_reference(problem,rank_value,common_seed,settings); % Compute the theorem-motivated initial gradient scale once for this matched run.
                for multiplier_index = 1:numel(settings.gamma_multipliers) % Change only the fixed-gamma multiplier.
                    gamma_multiplier = settings.gamma_multipliers(multiplier_index); % Read the current nonnegative multiplier.
                    gamma_value = gamma_multiplier*gamma_reference; % Set gamma=c*gamma_reference while holding all other variables fixed.
                    run_settings = settings; % Copy the common settings for this controlled run.
                    run_settings.gamma0 = gamma_value; % Supply the selected fixed consistency penalty to the existing aBMF solver.
                    fprintf('  r=%d, trial=%d, c=%.4g, gamma_ref=%.4g, gamma=%.4g ... ',rank_value,trial_index,gamma_multiplier,gamma_reference,gamma_value); % Display the current controlled run.
                    timer_value = tic; % Start measuring wall-clock time.
                    solution = solve_abmf_nc(problem,rank_value,common_seed,'aBMF-fixed',run_settings); % Solve the same asymmetric model with fixed gamma.
                    elapsed_time = toc(timer_value); % Obtain the wall-clock time.
                    metrics = evaluate_nc_solution(problem,solution,run_settings); % Evaluate the solution using the common metric routine.
                    row_counter = row_counter + 1; % Increase the completed-run counter.
                    raw_results(row_counter,1) = sensitivity_result_template(); % Create a result row with a fixed field layout.
                    raw_results(row_counter).sample_index = sample_index; % Store the selected test-set index.
                    raw_results(row_counter).epsilon = epsilon_value; % Store the perturbation radius.
                    raw_results(row_counter).rank = rank_value; % Store the factor rank.
                    raw_results(row_counter).trial = trial_index; % Store the matched trial index.
                    raw_results(row_counter).gamma_multiplier = gamma_multiplier; % Store the controlled multiplier c.
                    raw_results(row_counter).gamma_reference = gamma_reference; % Store the initial theorem-motivated gradient scale.
                    raw_results(row_counter).gamma_value = gamma_value; % Store the actual fixed consistency penalty.
                    raw_results(row_counter).iterations = solution.iterations; % Store the completed iteration number.
                    raw_results(row_counter).time = elapsed_time; % Store the wall-clock time.
                    raw_results(row_counter).surrogate_objective = metrics.surrogate_objective; % Store the complete nonconvex surrogate objective without the agreement penalty.
                    raw_results(row_counter).margin_surrogate = metrics.margin_surrogate; % Store the signed verification-margin surrogate.
                    raw_results(row_counter).affine_residual = metrics.affine_residual; % Store the maximum affine residual.
                    raw_results(row_counter).inequality_residual = metrics.inequality_residual; % Store the maximum inequality violation.
                    raw_results(row_counter).complementarity_residual = metrics.complementarity_residual; % Store the ReLU complementarity residual.
                    raw_results(row_counter).rel_disagreement = metrics.rel_disagreement; % Store the relative factor disagreement.
                    raw_results(row_counter).psd_violation = metrics.psd_violation; % Store the negative-eigenvalue PSD violation.
                    raw_results(row_counter).success = metrics.success; % Store the common diagnostic success flag.
                    fprintf('obj=%.5g, comp=%.3e, gap=%.3e, iter=%d, time=%.3fs.\n',metrics.surrogate_objective,metrics.complementarity_residual,metrics.rel_disagreement,solution.iterations,elapsed_time); % Display the main diagnostics.
                end % Finish the multiplier loop.
            end % Finish the trial loop.
        end % Finish the rank loop.
    end % Finish the perturbation-radius loop.
end % Finish the selected-sample loop.
condition_summary = summarize_gamma_sensitivity_by_condition(raw_results); % Aggregate means and standard deviations within each controlled condition.
aggregate = summarize_gamma_sensitivity_overall(raw_results,settings.gamma_multipliers); % Aggregate the complete experiment by multiplier.
matched_summary = summarize_gamma_sensitivity_matched(raw_results,settings.gamma_multipliers); % Compute matched changes relative to c=1.
write_gamma_sensitivity_raw_csv(raw_results,fullfile(settings.output_dir,'gamma_sensitivity_raw.csv')); % Write all individual controlled runs.
write_gamma_sensitivity_condition_csv(condition_summary,fullfile(settings.output_dir,'gamma_sensitivity_by_condition.csv')); % Write condition-level means and standard deviations.
write_gamma_sensitivity_aggregate_csv(aggregate,fullfile(settings.output_dir,'gamma_sensitivity_aggregate.csv')); % Write the compact multiplier-level table.
write_gamma_sensitivity_matched_csv(matched_summary,fullfile(settings.output_dir,'gamma_sensitivity_matched_vs_c1.csv')); % Write matched relative changes against the c=1 reference.
save(fullfile(settings.output_dir,'gamma_sensitivity_results.mat'),'raw_results','condition_summary','aggregate','matched_summary','settings'); % Save complete MATLAB result structures.
plot_gamma_sensitivity_aggregate(aggregate,settings.output_dir); % Generate four separate sensitivity figures.
fprintf('\nGamma multiplier sensitivity experiment completed. Results were saved to:\n%s\n',settings.output_dir); % Report successful completion and the output directory.
display_gamma_sensitivity_aggregate(aggregate,matched_summary); % Display the compact summaries in the command window.
end % Finish the controlled gamma-multiplier benchmark.

function settings = sensitivity_default_settings(run_mode,root_dir) % Construct the default settings for the controlled multiplier experiment.
settings.data_file = fullfile(root_dir,'data','digits_3_8_prepared.mat'); % Specify the bundled prepared Digits dataset and network.
settings.output_dir = fullfile(root_dir,'results_gamma_sensitivity'); % Specify a separate sensitivity-output directory.
settings.sample_positions = 1:3; % Use the same three margin-quantile test samples as the main benchmark.
settings.epsilon_list = [0.6,0.8]; % Use the same two infinity-norm perturbation radii as the main benchmark.
settings.rank_list = [8,16]; % Use the same two factor ranks as the main benchmark.
settings.num_trials = 10; % Repeat every randomized condition ten times.
settings.gamma_multipliers = [1,2,4,8,16]; % Test controlled overestimation factors c in gamma=c*gamma_reference.
settings.max_iter = 3000; % Use the same maximum iteration count as the main benchmark.
settings.tol = 1e-6; % Use the same relative factor-change stopping tolerance.
settings.initial_step = 1e-3; % Use the same initial gradient step for every block.
settings.maximum_step = 1e-2; % Use the same accepted-step cap.
settings.armijo = 1e-4; % Use the same sufficient-decrease coefficient.
settings.backtracking_factor = 0.5; % Use the same geometric backtracking reduction.
settings.maximum_backtracking = 25; % Use the same maximum number of line-search reductions.
settings.factor_norm_cap = 200; % Use the same numerical factor-norm cap.
settings.rho_eq = 50; % Use the same affine-equality penalty weight.
settings.rho_ineq = 50; % Use the same one-sided inequality penalty weight.
settings.rho_comp = 100; % Use the same nonconvex ReLU complementarity penalty weight.
settings.lambda_fro = 1e-3; % Use the same lifted-matrix Frobenius regularizer.
settings.gamma0 = NaN; % Initialize the fixed gamma placeholder before the multiplier-specific assignment.
settings.gamma_epsilon = 1e-6; % Add the same strict epsilon to the initial gradient-scale reference.
settings.gamma_max = 1e4; % Retain the existing solver field although the fixed-gamma method does not use it.
settings.epsilon_tau = 1e-12; % Retain the existing solver field although the fixed-gamma method does not use it.
settings.seed_base = 314159; % Use the same common random-seed base as the main benchmark.
settings.success_tolerance = 5e-2; % Use the same common diagnostic residual threshold.
settings.save_histories = false; % Disable the main-benchmark history logic for this controlled sweep.
settings.history_sample_position = 2; % Retain the existing settings field for compatibility.
settings.history_epsilon_position = 2; % Retain the existing settings field for compatibility.
if strcmpi(run_mode,'quick') % Check whether the user requested a smoke test.
    settings.sample_positions = 2; % Use only the median-margin sample.
    settings.epsilon_list = 0.8; % Use one perturbation radius.
    settings.rank_list = 8; % Use one factor rank.
    settings.num_trials = 1; % Use one randomized trial.
    settings.gamma_multipliers = [1,2,4]; % Use three multipliers in the smoke test.
    settings.max_iter = 80; % Use a short iteration limit for the smoke test.
elseif ~strcmpi(run_mode,'full') % Reject unsupported run-mode names.
    error('Unknown run mode: %s. Use ''full'' or ''quick''.',run_mode); % Stop for an unsupported run mode.
end % Finish the run-mode selection.
end % Finish the controlled-setting constructor.

function gamma_reference = compute_initial_gamma_reference(problem,rank_value,seed_value,settings) % Compute the initial theorem-motivated gradient scale with matched factor initialization.
rng(seed_value,'twister'); % Set the deterministic matched initialization seed.
X = zeros(problem.p,rank_value); % Allocate the initial factor.
X(:,1) = problem.nominal_vector; % Place the nominal rank-one lift in the first factor column.
if rank_value > 1 % Check whether additional factor columns are available.
    X(:,2:rank_value) = 1e-3*randn(problem.p,rank_value-1); % Add the same small Gaussian perturbation columns used by solve_abmf_nc.
end % Finish the additional-column initialization.
Z0 = X*X.'; % Form the initial symmetric lifted matrix because X0=Y0.
[~,~,parts] = evaluate_nonconvex_lift(problem,Z0); % Evaluate the initial nonconvex lifted gradient scale.
gamma_reference = 0.5*parts.gradient_fro + settings.gamma_epsilon; % Define gamma_reference=0.5*norm(gradient_Z,'fro')+epsilon_gamma.
if ~isfinite(gamma_reference) || gamma_reference <= 0 % Check whether the computed reference is numerically valid.
    error('The initial gamma reference is invalid for rank %d and seed %d.',rank_value,seed_value); % Stop before using an invalid sensitivity scale.
end % Finish the reference-validity check.
end % Finish the initial gamma-reference computation.

function row = sensitivity_result_template() % Create one fixed-layout raw-result structure.
row = struct('sample_index',0,'epsilon',NaN,'rank',0,'trial',0,'gamma_multiplier',NaN,'gamma_reference',NaN,'gamma_value',NaN,'iterations',0,'time',NaN,'surrogate_objective',NaN,'margin_surrogate',NaN,'affine_residual',NaN,'inequality_residual',NaN,'complementarity_residual',NaN,'rel_disagreement',NaN,'psd_violation',NaN,'success',0); % Define all raw-result fields in a stable order.
end % Finish the raw-result template.
