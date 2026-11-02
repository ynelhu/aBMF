function solution = solve_abmf_nc(problem,rank_value,seed_value,method_name,settings) % Solve one asymmetric low-rank nonconvex verification surrogate.
[X,Y] = initialize_asymmetric_factors(problem,rank_value,seed_value); % Construct the matched nominal initialization.
gamma_value = settings.gamma0; % Use the common initial consistency penalty.
[complete_objective,base_objective,gradient_Z,parts] = evaluate_asymmetric_objective(problem,X,Y,gamma_value); % Evaluate the initial asymmetric objective.
if strcmp(method_name,'aBMF-our') % Apply the theorem-motivated safeguard before the first factor update.
    gamma_hat = 0.5*parts.gradient_fro + settings.gamma_epsilon; % Form the strict gradient-norm candidate at the initial lifted point.
    if ~isfinite(gamma_hat) % Check whether the initial candidate is numerically valid.
        error('The initial aBMF-our gamma candidate is not finite.'); % Stop before using an invalid consistency penalty.
    end % Finish the initial-candidate validity check.
    gamma_value = max(gamma_value,gamma_hat); % Enforce gamma_0=max(gamma_initial,0.5*gradient_fro+epsilon).
    [complete_objective,base_objective,gradient_Z,parts] = evaluate_asymmetric_objective(problem,X,Y,gamma_value); % Re-evaluate the complete objective with the safeguarded initial gamma.
end % Finish the initial aBMF-our safeguard.
step_X = settings.initial_step; % Initialize the X-block gradient step.
step_Y = settings.initial_step; % Initialize the Y-block gradient step.
history = initialize_asymmetric_history(settings.max_iter); % Allocate fixed-size history arrays.
completed_iterations = 0; % Initialize the completed-iteration counter.
for iteration = 1:settings.max_iter % Perform alternating factor-gradient iterations.
    previous_X = X; % Store the previous X factor for the stopping test.
    previous_Y = Y; % Store the previous Y factor for the stopping test.
    gradient_X = gradient_Z*Y + gamma_value*(X-Y); % Compute the X-block gradient for the symmetrized split.
    gradient_X_norm_squared = sum(gradient_X(:).^2); % Compute the squared X-block gradient norm.
    [X,complete_objective,base_objective,gradient_Z,parts,step_X,accepted_X] = asymmetric_block_step(problem,X,Y,gamma_value,gradient_X,gradient_X_norm_squared,complete_objective,step_X,'X',settings); % Perform one safeguarded X-block step.
    gradient_Y = gradient_Z*X + gamma_value*(Y-X); % Compute the Y-block gradient after the accepted X update.
    gradient_Y_norm_squared = sum(gradient_Y(:).^2); % Compute the squared Y-block gradient norm.
    [Y,complete_objective,base_objective,gradient_Z,parts,step_Y,accepted_Y] = asymmetric_block_step(problem,X,Y,gamma_value,gradient_Y,gradient_Y_norm_squared,complete_objective,step_Y,'Y',settings); % Perform one safeguarded Y-block step.
    gamma_hat = 0.5*parts.gradient_fro + settings.gamma_epsilon; % Form the strict theorem-motivated gradient-norm candidate.
    if strcmp(method_name,'aBMF-our') % Apply the nondecreasing spectral-safeguard rule for the nonconvex stress test.
        if ~isfinite(gamma_hat) % Check whether the current candidate is numerically valid.
            error('The aBMF-our gamma candidate is not finite at iteration %d.',iteration); % Stop before using an invalid consistency penalty.
        end % Finish the current-candidate validity check.
        gamma_value = max(gamma_hat,gamma_value); % Update gamma_k=max(0.5*gradient_fro+epsilon,gamma_{k-1}).
    elseif strcmp(method_name,'aBMF-adp') % Apply the multiplicative adaptive baseline.
        numerator = sum(X(:).^2)+sum(Y(:).^2); % Compute the multiplicative-update numerator.
        denominator = 2*max(abs(sum(X(:).*Y(:))),settings.epsilon_tau); % Compute the safeguarded multiplicative-update denominator.
        multiplier = max(1,numerator/denominator); % Construct the nondecreasing multiplicative factor.
        gamma_value = min(settings.gamma_max,max(settings.gamma_epsilon,gamma_value*multiplier)); % Apply the safeguarded multiplicative update.
    elseif strcmp(method_name,'aBMF-fixed') % Keep the common initial penalty fixed.
        gamma_value = gamma_value; % Preserve the fixed consistency penalty explicitly.
    else % Reject unsupported asymmetric method names.
        error('Unknown asymmetric method: %s',method_name); % Stop for an unsupported method.
    end % Finish the gamma-update selection.
    [complete_objective,base_objective,gradient_Z,parts] = evaluate_asymmetric_objective(problem,X,Y,gamma_value); % Re-evaluate the asymmetric objective after changing gamma.
    relative_X = norm(X-previous_X,'fro')/max(1,norm(previous_X,'fro')); % Compute the relative X-factor change.
    relative_Y = norm(Y-previous_Y,'fro')/max(1,norm(previous_Y,'fro')); % Compute the relative Y-factor change.
    relative_disagreement = norm(X-Y,'fro')/max([1,norm(X,'fro'),norm(Y,'fro')]); % Compute the relative factor disagreement.
    completed_iterations = iteration; % Store the completed iteration number.
    history.iteration(iteration) = iteration; % Store the iteration index.
    history.objective(iteration) = base_objective; % Store the nonconvex lifted surrogate objective without the agreement penalty.
    history.complete_objective(iteration) = complete_objective; % Store the complete asymmetric objective including the agreement penalty.
    history.margin(iteration) = parts.margin; % Store the signed margin surrogate.
    history.affine(iteration) = parts.affine_residual; % Store the affine residual.
    history.inequality(iteration) = parts.inequality_residual; % Store the inequality residual.
    history.complementarity(iteration) = parts.complementarity_residual; % Store the complementarity residual.
    history.disagreement(iteration) = relative_disagreement; % Store the relative factor disagreement.
    history.gamma(iteration) = gamma_value; % Store the current consistency penalty.
    history.gamma_hat(iteration) = gamma_hat; % Store the strict gradient-based candidate.
    if (~accepted_X && ~accepted_Y) || max(relative_X,relative_Y) <= settings.tol % Check line-search failure or factor stabilization.
        break; % Stop the asymmetric method.
    end % Finish the stopping test.
end % Finish the asymmetric iterations.
solution.X = X; % Store the final first factor.
solution.Y = Y; % Store the final second factor.
solution.iterations = completed_iterations; % Store the completed iteration count.
solution.final_gamma = gamma_value; % Store the final consistency penalty.
solution.final_gamma_hat = 0.5*parts.gradient_fro + settings.gamma_epsilon; % Store the final strict gradient-based candidate.
solution.history = trim_asymmetric_history(history,completed_iterations); % Remove unused trailing history entries.
end % Finish the asymmetric nonconvex solver.

function [X,Y] = initialize_asymmetric_factors(problem,rank_value,seed_value) % Construct identical matched initial factors.
rng(seed_value,'twister'); % Set the deterministic initialization seed.
X = zeros(problem.p,rank_value); % Allocate the first factor.
X(:,1) = problem.nominal_vector; % Place the nominal rank-one lift in the first factor column.
if rank_value > 1 % Check whether additional factor columns are available.
    X(:,2:rank_value) = 1e-3*randn(problem.p,rank_value-1); % Add small shared Gaussian perturbation columns.
end % Finish the additional-column initialization.
Y = X; % Use an identical second factor at initialization.
end % Finish the asymmetric initialization.

function [complete_objective,base_objective,gradient_Z,parts] = evaluate_asymmetric_objective(problem,X,Y,gamma_value) % Evaluate the complete asymmetric objective and lifted gradient.
Z = 0.5*(X*Y.'+Y*X.'); % Form the symmetric matrix represented by the asymmetric factors.
[base_objective,gradient_Z,parts] = evaluate_nonconvex_lift(problem,Z); % Evaluate the genuinely nonconvex lifted surrogate.
agreement_penalty = 0.5*gamma_value*sum((X(:)-Y(:)).^2); % Evaluate the quadratic factor-consistency penalty.
complete_objective = base_objective + agreement_penalty; % Form the complete asymmetric objective.
end % Finish asymmetric objective evaluation.

function [updated_factor,complete_objective,base_objective,gradient_Z,parts,next_step,accepted] = asymmetric_block_step(problem,X,Y,gamma_value,block_gradient,gradient_norm_squared,current_objective,current_step,block_name,settings) % Perform one Armijo-protected asymmetric block-gradient step.
trial_step = current_step; % Initialize the current line-search step.
accepted = false; % Initialize the acceptance flag.
if strcmp(block_name,'X') % Select the unchanged factor returned after an X-block line-search failure.
    updated_factor = X; % Preserve the current X factor safely.
else % Handle a Y-block line-search failure.
    updated_factor = Y; % Preserve the current Y factor safely.
end % Finish the unchanged-factor selection.
complete_objective = current_objective; % Initialize the complete objective safely.
base_objective = NaN; % Initialize the base objective safely.
gradient_Z = zeros(problem.p,problem.p); % Initialize the lifted gradient safely.
parts = struct('margin',NaN,'affine_residual',NaN,'inequality_residual',NaN,'complementarity_residual',NaN,'objective',NaN,'gradient_fro',NaN); % Initialize diagnostic parts safely.
for backtracking_index = 1:settings.maximum_backtracking % Perform geometric Armijo backtracking.
    if strcmp(block_name,'X') % Check whether the X block is being updated.
        candidate_X = cap_asymmetric_factor(X-trial_step*block_gradient,settings.factor_norm_cap); % Construct the safeguarded candidate X factor.
        candidate_Y = Y; % Keep the Y factor fixed.
        candidate_factor = candidate_X; % Define the returned updated factor.
    else % Handle a Y-block update.
        candidate_X = X; % Keep the X factor fixed.
        candidate_Y = cap_asymmetric_factor(Y-trial_step*block_gradient,settings.factor_norm_cap); % Construct the safeguarded candidate Y factor.
        candidate_factor = candidate_Y; % Define the returned updated factor.
    end % Finish the block selection.
    [candidate_complete,candidate_base,candidate_gradient_Z,candidate_parts] = evaluate_asymmetric_objective(problem,candidate_X,candidate_Y,gamma_value); % Evaluate the candidate asymmetric objective.
    if isfinite(candidate_complete) && candidate_complete <= current_objective-settings.armijo*trial_step*gradient_norm_squared % Test the Armijo sufficient-decrease condition.
        accepted = true; % Record successful line-search acceptance.
        updated_factor = candidate_factor; % Accept the updated block factor.
        complete_objective = candidate_complete; % Accept the complete asymmetric objective.
        base_objective = candidate_base; % Accept the nonconvex lifted objective.
        gradient_Z = candidate_gradient_Z; % Accept the lifted gradient.
        parts = candidate_parts; % Accept the diagnostic parts.
        break; % Finish the line search.
    end % Finish the Armijo test.
    trial_step = settings.backtracking_factor*trial_step; % Reduce the trial step geometrically.
end % Finish the backtracking loop.
if accepted % Check whether the block step was accepted.
    next_step = min(settings.maximum_step,1.2*trial_step); % Mildly enlarge the next initial trial step.
else % Handle line-search failure.
    next_step = max(settings.gamma_epsilon,trial_step); % Retain a small positive next step.
    [complete_objective,base_objective,gradient_Z,parts] = evaluate_asymmetric_objective(problem,X,Y,gamma_value); % Restore diagnostics at the unchanged factors.
end % Finish the acceptance-dependent step update.
end % Finish the asymmetric block-step helper.

function factor = cap_asymmetric_factor(factor,norm_cap) % Cap one asymmetric factor Frobenius norm.
factor_norm = norm(factor,'fro'); % Compute the factor Frobenius norm.
if factor_norm > norm_cap % Check whether the numerical cap is exceeded.
    factor = factor*(norm_cap/factor_norm); % Scale the factor back to the permitted norm.
end % Finish the norm-cap check.
end % Finish the asymmetric factor safeguard.

function history = initialize_asymmetric_history(max_iter) % Allocate fixed-size asymmetric history arrays.
history.iteration = zeros(max_iter,1); % Allocate iteration indices.
history.objective = NaN(max_iter,1); % Allocate lifted surrogate objectives.
history.complete_objective = NaN(max_iter,1); % Allocate complete asymmetric objectives.
history.margin = NaN(max_iter,1); % Allocate signed margin surrogates.
history.affine = NaN(max_iter,1); % Allocate affine residuals.
history.inequality = NaN(max_iter,1); % Allocate inequality residuals.
history.complementarity = NaN(max_iter,1); % Allocate complementarity residuals.
history.disagreement = NaN(max_iter,1); % Allocate factor disagreements.
history.gamma = NaN(max_iter,1); % Allocate consistency penalties.
history.gamma_hat = NaN(max_iter,1); % Allocate strict gradient-based penalty candidates.
end % Finish asymmetric history allocation.

function history = trim_asymmetric_history(history,completed_iterations) % Remove unused trailing asymmetric history entries.
field_list = fieldnames(history); % Obtain all asymmetric history field names.
for field_index = 1:numel(field_list) % Trim every history vector consistently.
    field_name = field_list{field_index}; % Read the current history field name.
    history.(field_name) = history.(field_name)(1:completed_iterations); % Retain only completed iterations.
end % Finish trimming all asymmetric history fields.
end % Finish the asymmetric history-trimming helper.
