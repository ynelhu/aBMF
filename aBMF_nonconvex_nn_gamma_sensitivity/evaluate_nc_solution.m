function metrics = evaluate_nc_solution(problem,solution,settings) % Evaluate one symmetric or asymmetric nonconvex verification solution.
Z = 0.5*(solution.X*solution.Y.'+solution.Y*solution.X.'); % Form the common symmetrized lifted matrix.
[objective_value,~,parts] = evaluate_nonconvex_lift(problem,Z); % Evaluate the nonconvex lifted surrogate and residual parts.
metrics.surrogate_objective = objective_value; % Store the complete lifted surrogate objective without the agreement penalty.
metrics.margin_surrogate = parts.margin; % Store the signed verification-margin surrogate.
metrics.affine_residual = parts.affine_residual; % Store the maximum affine equality residual.
metrics.inequality_residual = parts.inequality_residual; % Store the maximum inequality violation.
metrics.complementarity_residual = parts.complementarity_residual; % Store the ReLU complementarity residual.
metrics.rel_disagreement = norm(solution.X-solution.Y,'fro')/max([1,norm(solution.X,'fro'),norm(solution.Y,'fro')]); % Compute the relative factor disagreement.
minimum_eigenvalue = min(real(eig(0.5*(Z+Z.')))); % Compute the minimum eigenvalue of the symmetrized split.
metrics.psd_violation = max(0,-minimum_eigenvalue); % Store the negative-eigenvalue PSD violation.
diagnostic_maximum = max([metrics.affine_residual,metrics.inequality_residual,metrics.complementarity_residual,metrics.rel_disagreement]); % Form the common diagnostic residual maximum.
metrics.success = double(isfinite(objective_value) && diagnostic_maximum <= settings.success_tolerance); % Define one common finite-residual success flag.
end % Finish common solution evaluation.
