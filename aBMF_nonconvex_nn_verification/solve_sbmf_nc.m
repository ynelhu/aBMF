function solution = solve_sbmf_nc(problem,rank_value,seed_value,settings) % Solve the symmetric low-rank nonconvex verification surrogate.
[X,~] = initialize_factors(problem,rank_value,seed_value); % Construct the common nominal low-rank initialization.
Z = X*X.'; % Construct the initial symmetric PSD lift.
[objective_value,gradient_Z,parts] = evaluate_nonconvex_lift(problem,Z); % Evaluate the initial nonconvex objective and gradient.
step_length = settings.initial_step; % Initialize the gradient step length.
history = initialize_history(settings.max_iter); % Allocate a fixed-size optimization history.
completed_iterations = 0; % Initialize the completed-iteration counter.
for iteration = 1:settings.max_iter % Perform symmetric factor-gradient iterations.
    gradient_X = 2*gradient_Z*X; % Apply the chain rule for Z=XX'.
    gradient_norm_squared = sum(gradient_X(:).^2); % Compute the squared factor-gradient norm.
    if gradient_norm_squared <= settings.tol^2 % Check first-order numerical stationarity.
        break; % Stop after reaching the gradient tolerance.
    end % Finish the stationarity check.
    trial_step = step_length; % Initialize the current line-search step.
    accepted = false; % Initialize the line-search acceptance flag.
    for backtracking_index = 1:settings.maximum_backtracking % Perform Armijo backtracking.
        candidate_X = X-trial_step*gradient_X; % Take a symmetric factor-gradient step.
        candidate_X = cap_factor_norm(candidate_X,settings.factor_norm_cap); % Protect against numerical factor explosion.
        candidate_Z = candidate_X*candidate_X.'; % Construct the candidate symmetric PSD lift.
        [candidate_objective,candidate_gradient_Z,candidate_parts] = evaluate_nonconvex_lift(problem,candidate_Z); % Evaluate the candidate nonconvex objective.
        if isfinite(candidate_objective) && candidate_objective <= objective_value-settings.armijo*trial_step*gradient_norm_squared % Test the Armijo sufficient-decrease condition.
            accepted = true; % Record successful line-search acceptance.
            break; % Finish the backtracking search.
        end % Finish the Armijo test.
        trial_step = settings.backtracking_factor*trial_step; % Reduce the trial step geometrically.
    end % Finish the backtracking loop.
    if ~accepted % Check whether no acceptable step was found.
        break; % Stop the method after line-search failure.
    end % Finish the failed-search check.
    relative_change = norm(candidate_X-X,'fro')/max(1,norm(X,'fro')); % Compute the relative factor change.
    X = candidate_X; % Accept the candidate factor.
    objective_value = candidate_objective; % Accept the candidate objective.
    gradient_Z = candidate_gradient_Z; % Accept the candidate lifted gradient.
    parts = candidate_parts; % Accept the candidate diagnostic parts.
    step_length = min(settings.maximum_step,1.2*trial_step); % Mildly enlarge the next initial trial step.
    completed_iterations = iteration; % Store the completed iteration number.
    history.iteration(iteration) = iteration; % Store the iteration number.
    history.objective(iteration) = objective_value; % Store the nonconvex surrogate objective.
    history.margin(iteration) = parts.margin; % Store the signed margin surrogate.
    history.affine(iteration) = parts.affine_residual; % Store the affine residual.
    history.inequality(iteration) = parts.inequality_residual; % Store the inequality residual.
    history.complementarity(iteration) = parts.complementarity_residual; % Store the complementarity residual.
    history.disagreement(iteration) = 0; % Store zero disagreement for the symmetric method.
    history.gamma(iteration) = NaN; % Mark gamma as unavailable for the symmetric method.
    history.gamma_hat(iteration) = 0.5*parts.gradient_fro + settings.gamma_epsilon; % Store the strict gradient-based candidate for reference.
    if relative_change <= settings.tol % Check factor stabilization.
        break; % Stop after the common relative-change criterion is met.
    end % Finish the stabilization check.
end % Finish the symmetric iterations.
solution.X = X; % Store the final symmetric factor.
solution.Y = X; % Store the same factor as the second factor for common evaluation.
solution.iterations = completed_iterations; % Store the completed iteration count.
solution.final_gamma = NaN; % Mark the consistency penalty as unavailable.
solution.final_gamma_hat = 0.5*parts.gradient_fro + settings.gamma_epsilon; % Store the final strict gradient-based candidate.
solution.history = trim_history(history,completed_iterations); % Remove unused trailing history entries.
end % Finish the symmetric nonconvex solver.

function [X,Y] = initialize_factors(problem,rank_value,seed_value) % Construct a matched nominal initialization for both factorization families.
rng(seed_value,'twister'); % Set the deterministic initialization seed.
X = zeros(problem.p,rank_value); % Allocate the first factor.
X(:,1) = problem.nominal_vector; % Place the nominal rank-one lift in the first factor column.
if rank_value > 1 % Check whether additional factor columns are available.
    X(:,2:rank_value) = 1e-3*randn(problem.p,rank_value-1); % Add small shared Gaussian perturbation columns.
end % Finish the extra-column initialization.
Y = X; % Use identical initial factors for every asymmetric method.
end % Finish the common factor initialization.

function factor = cap_factor_norm(factor,norm_cap) % Cap a factor Frobenius norm without changing its direction.
factor_norm = norm(factor,'fro'); % Compute the current factor Frobenius norm.
if factor_norm > norm_cap % Check whether the numerical norm cap is exceeded.
    factor = factor*(norm_cap/factor_norm); % Scale the factor back to the permitted norm.
end % Finish the norm-cap check.
end % Finish the factor-norm safeguard.

function history = initialize_history(max_iter) % Allocate fixed-size history arrays.
history.iteration = zeros(max_iter,1); % Allocate iteration indices.
history.objective = NaN(max_iter,1); % Allocate nonconvex objective values.
history.margin = NaN(max_iter,1); % Allocate signed margin surrogates.
history.affine = NaN(max_iter,1); % Allocate affine residuals.
history.inequality = NaN(max_iter,1); % Allocate inequality residuals.
history.complementarity = NaN(max_iter,1); % Allocate complementarity residuals.
history.disagreement = NaN(max_iter,1); % Allocate relative factor disagreements.
history.gamma = NaN(max_iter,1); % Allocate consistency penalties.
history.gamma_hat = NaN(max_iter,1); % Allocate strict gradient-based penalty candidates.
end % Finish the history allocation.

function history = trim_history(history,completed_iterations) % Remove unused trailing history entries.
field_list = fieldnames(history); % Obtain all history field names.
for field_index = 1:numel(field_list) % Trim every history vector consistently.
    field_name = field_list{field_index}; % Read the current history field name.
    history.(field_name) = history.(field_name)(1:completed_iterations); % Retain only completed iterations.
end % Finish trimming all history fields.
end % Finish the history-trimming helper.
