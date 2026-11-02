function self_check() % Check the bundled data, lifted objective, and analytical matrix gradient.
root_dir = fileparts(mfilename('fullpath')); % Locate the project directory.
dataset = load_digits_network(fullfile(root_dir,'data','digits_3_8_prepared.mat')); % Load the bundled fixed network and data.
settings.rho_eq = 50; % Set the affine-equality penalty weight used by the check.
settings.rho_ineq = 50; % Set the inequality penalty weight used by the check.
settings.rho_comp = 100; % Set the complementarity penalty weight used by the check.
settings.lambda_fro = 1e-3; % Set the lifted Frobenius regularization weight used by the check.
sample_index = dataset.selected_test_indices(2); % Select the median-margin verification sample.
problem = build_verification_problem(dataset,dataset.X_test(:,sample_index),dataset.y_test(sample_index),0.8,settings); % Construct one representative verification problem.
Z = problem.nominal_vector*problem.nominal_vector.'; % Construct the nominal rank-one lifted matrix.
[objective_value,gradient_Z,parts] = evaluate_nonconvex_lift(problem,Z); % Evaluate the nominal nonconvex objective and gradient.
rng(123,'twister'); % Set the deterministic finite-difference seed.
direction = randn(problem.p,problem.p); % Generate a random matrix perturbation.
direction = 0.5*(direction+direction.'); % Restrict the perturbation to the symmetric matrix space.
step_value = 1e-6; % Set the centered finite-difference step.
objective_plus = evaluate_nonconvex_lift(problem,Z+step_value*direction); % Evaluate the positive finite-difference point.
objective_minus = evaluate_nonconvex_lift(problem,Z-step_value*direction); % Evaluate the negative finite-difference point.
finite_difference = (objective_plus-objective_minus)/(2*step_value); % Compute the centered directional derivative.
analytic_derivative = sum(sum(gradient_Z.*direction)); % Compute the analytical directional derivative.
relative_error = abs(finite_difference-analytic_derivative)/max([1,abs(finite_difference),abs(analytic_derivative)]); % Compute a scale-independent derivative error.
fprintf('Network test accuracy: %.6f.\n',dataset.test_accuracy); % Display the bundled network accuracy.
fprintf('Nominal objective: %.6g.\n',objective_value); % Display the nominal surrogate objective.
fprintf('Nominal affine residual: %.3e.\n',parts.affine_residual); % Display the nominal affine residual.
fprintf('Nominal inequality residual: %.3e.\n',parts.inequality_residual); % Display the nominal inequality residual.
fprintf('Nominal complementarity residual: %.3e.\n',parts.complementarity_residual); % Display the nominal complementarity residual.
fprintf('Gradient finite-difference relative error: %.3e.\n',relative_error); % Display the matrix-gradient consistency error.
if relative_error > 5e-3 % Check whether the analytical gradient is inconsistent with finite differences.
    error('The lifted-gradient self-check failed.'); % Stop when the derivative error is too large.
end % Finish the gradient-error check.
solver_settings = settings; % Start the solver smoke-test settings from the objective weights.
solver_settings.max_iter = 3; % Limit the asymmetric smoke test to three iterations.
solver_settings.tol = 0; % Prevent an early tolerance stop during the short smoke test.
solver_settings.initial_step = 1e-3; % Set the initial block-gradient step.
solver_settings.maximum_step = 1e-2; % Cap the block-gradient step.
solver_settings.armijo = 1e-4; % Set the Armijo sufficient-decrease coefficient.
solver_settings.backtracking_factor = 0.5; % Set the geometric backtracking factor.
solver_settings.maximum_backtracking = 25; % Limit the number of backtracking reductions.
solver_settings.factor_norm_cap = 200; % Set the factor-norm numerical safeguard.
solver_settings.gamma0 = 10; % Set the common initial consistency penalty.
solver_settings.gamma_epsilon = 1e-6; % Set the strict gamma safeguard.
solver_settings.gamma_max = 1e4; % Set the multiplicative-baseline cap required by the shared solver.
solver_settings.epsilon_tau = 1e-12; % Set the multiplicative-baseline denominator safeguard.
solution = solve_abmf_nc(problem,4,123,'aBMF-our',solver_settings); % Run a short aBMF-our solver smoke test.
if any(diff(solution.history.gamma) < -1e-12) % Check whether the proposed gamma sequence ever decreases.
    error('The aBMF-our gamma sequence is not nondecreasing.'); % Stop when the updated rule is violated.
end % Finish the gamma-monotonicity check.
if any(solution.history.gamma+1e-12 < solution.history.gamma_hat) % Check whether every stored gamma dominates its strict candidate.
    error('The aBMF-our gamma safeguard is not satisfied.'); % Stop when gamma falls below the gradient-based candidate.
end % Finish the gamma-safeguard check.
fprintf('aBMF-our smoke-test final gamma: %.6g.\n',solution.final_gamma); % Display the final smoke-test penalty.
fprintf('Self-check passed.\n'); % Report successful completion.
end % Finish the project self-check.
