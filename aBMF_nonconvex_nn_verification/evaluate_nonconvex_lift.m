function [objective_value,gradient_Z,parts] = evaluate_nonconvex_lift(problem,Z) % Evaluate the nonconvex lifted loss and its symmetric matrix gradient.
Z = 0.5*(Z+Z.'); % Enforce numerical symmetry before evaluating the lifted objective.
p = problem.p; % Read the lifted matrix dimension.
index_x = problem.index_x; % Read the lifted input-coordinate indices.
index_a = problem.index_a; % Read the lifted activation-coordinate indices.
z00 = Z(1,1); % Read the homogenizing first moment.
x = Z(1,index_x).'; % Read the lifted input first moments.
a = Z(1,index_a).'; % Read the lifted activation first moments.
s = problem.W1*x + problem.b1*z00; % Compute the lifted first-order preactivation vector.
gradient_Z = 2*problem.lambda_fro*Z; % Initialize the gradient with the Frobenius regularizer.
margin_value = problem.y0*(problem.w2.'*a + problem.b2*z00); % Evaluate the signed verification-margin surrogate.
objective_value = margin_value + problem.lambda_fro*sum(Z(:).^2); % Initialize the complete nonconvex surrogate objective.
gradient_Z = add_first_moment_gradient(gradient_Z,index_a,problem.y0*problem.w2); % Add the activation derivative of the margin.
gradient_Z(1,1) = gradient_Z(1,1) + problem.y0*problem.b2; % Add the homogenizing derivative of the margin.
affine_values = zeros(1+numel(a),1); % Allocate a safe upper bound for affine residual storage.
affine_count = 1; % Initialize the affine-residual counter.
affine_values(affine_count) = z00-1; % Store the homogenizing equality residual.
objective_value = objective_value + problem.rho_eq*(z00-1)^2; % Penalize violation of the homogenizing equality.
gradient_Z(1,1) = gradient_Z(1,1) + 2*problem.rho_eq*(z00-1); % Add the homogenizing equality derivative.
gradient_x = zeros(numel(x),1); % Allocate accumulated first-moment derivatives with respect to x.
gradient_a = zeros(numel(a),1); % Allocate accumulated first-moment derivatives with respect to a.
gradient_z00 = 0; % Initialize the accumulated homogenizing derivative.
for hidden_index = 1:numel(a) % Add affine equalities for provably stable ReLU units.
    if problem.upper_s(hidden_index) <= 0 % Check whether the ReLU is provably inactive.
        equality_value = a(hidden_index); % Define the inactive-ReLU equality a_i=0.
        affine_count = affine_count + 1; % Increase the affine-residual counter.
        affine_values(affine_count) = equality_value; % Store the inactive-ReLU residual.
        objective_value = objective_value + problem.rho_eq*equality_value^2; % Penalize the inactive-ReLU equality.
        gradient_a(hidden_index) = gradient_a(hidden_index) + 2*problem.rho_eq*equality_value; % Add the inactive-ReLU derivative.
    elseif problem.lower_s(hidden_index) >= 0 % Check whether the ReLU is provably active.
        equality_value = a(hidden_index)-s(hidden_index); % Define the active-ReLU equality a_i=s_i.
        affine_count = affine_count + 1; % Increase the affine-residual counter.
        affine_values(affine_count) = equality_value; % Store the active-ReLU residual.
        equality_coefficient = 2*problem.rho_eq*equality_value; % Compute the equality-penalty derivative coefficient.
        objective_value = objective_value + problem.rho_eq*equality_value^2; % Penalize the active-ReLU equality.
        gradient_a(hidden_index) = gradient_a(hidden_index) + equality_coefficient; % Add the activation derivative.
        gradient_x = gradient_x - equality_coefficient*problem.W1(hidden_index,:).'; % Add the input derivative through s_i.
        gradient_z00 = gradient_z00 - equality_coefficient*problem.b1(hidden_index); % Add the homogenizing derivative through s_i.
    end % Finish the stable-ReLU case distinction.
end % Finish the stable-ReLU equality loop.
gradient_Z = add_first_moment_gradient(gradient_Z,index_x,gradient_x); % Add all accumulated input first-moment derivatives.
gradient_Z = add_first_moment_gradient(gradient_Z,index_a,gradient_a); % Add all accumulated activation first-moment derivatives.
gradient_Z(1,1) = gradient_Z(1,1) + gradient_z00; % Add the accumulated homogenizing derivative.
inequality_values = zeros(4*numel(x)+4*numel(a),1); % Allocate a safe upper bound for inequality residual storage.
inequality_count = 0; % Initialize the inequality-residual counter.
for input_index = 1:numel(x) % Add input box and quadratic-envelope inequalities.
    value_lower = problem.lower_x(input_index)*z00-x(input_index); % Form the lower-box violation expression.
    [objective_value,gradient_Z,inequality_values,inequality_count] = add_positive_square(objective_value,gradient_Z,inequality_values,inequality_count,value_lower,problem.rho_ineq,index_x,input_index,-1,[],0,0,problem.lower_x(input_index),0,0); % Penalize lower-box violation.
    value_upper = x(input_index)-problem.upper_x(input_index)*z00; % Form the upper-box violation expression.
    [objective_value,gradient_Z,inequality_values,inequality_count] = add_positive_square(objective_value,gradient_Z,inequality_values,inequality_count,value_upper,problem.rho_ineq,index_x,input_index,1,[],0,0,-problem.upper_x(input_index),0,0); % Penalize upper-box violation.
    value_quadratic = Z(index_x(input_index),index_x(input_index))-(problem.lower_x(input_index)+problem.upper_x(input_index))*x(input_index)+problem.lower_x(input_index)*problem.upper_x(input_index)*z00; % Form the quadratic input-envelope violation.
    [objective_value,gradient_Z,inequality_values,inequality_count] = add_positive_square(objective_value,gradient_Z,inequality_values,inequality_count,value_quadratic,problem.rho_ineq,index_x,input_index,-(problem.lower_x(input_index)+problem.upper_x(input_index)),[],0,0,problem.lower_x(input_index)*problem.upper_x(input_index),index_x(input_index),1); % Penalize the quadratic input-envelope violation.
end % Finish the input-inequality loop.
for hidden_index = 1:numel(a) % Add ReLU envelope and activation-moment inequalities.
    value_nonnegative = -a(hidden_index); % Form the violation expression for a_i>=0.
    [objective_value,gradient_Z,inequality_values,inequality_count] = add_positive_square(objective_value,gradient_Z,inequality_values,inequality_count,value_nonnegative,problem.rho_ineq,[],0,0,index_a,hidden_index,-1,0,0,0); % Penalize negative activation first moments.
    value_above_s = s(hidden_index)-a(hidden_index); % Form the violation expression for a_i>=s_i.
    [objective_value,gradient_Z,inequality_values,inequality_count] = add_relu_difference_penalty(objective_value,gradient_Z,inequality_values,inequality_count,value_above_s,problem.rho_ineq,problem,hidden_index,-1,1); % Penalize violation of a_i>=s_i.
    if problem.lower_s(hidden_index) < 0 && problem.upper_s(hidden_index) > 0 % Check whether the ReLU is unstable.
        alpha_value = problem.upper_s(hidden_index)/(problem.upper_s(hidden_index)-problem.lower_s(hidden_index)); % Compute the convex-envelope slope.
        value_upper_relu = a(hidden_index)-alpha_value*s(hidden_index)+alpha_value*problem.lower_s(hidden_index)*z00; % Form the unstable-ReLU upper-envelope violation.
        [objective_value,gradient_Z,inequality_values,inequality_count] = add_relu_upper_penalty(objective_value,gradient_Z,inequality_values,inequality_count,value_upper_relu,problem.rho_ineq,problem,hidden_index,alpha_value); % Penalize the unstable-ReLU upper-envelope violation.
    end % Finish the unstable-ReLU check.
    if problem.upper_s(hidden_index) > 0 % Check whether a finite positive activation upper bound is available.
        value_activation_moment = Z(index_a(hidden_index),index_a(hidden_index))-problem.upper_s(hidden_index)*a(hidden_index); % Form the activation-moment upper-bound violation.
        [objective_value,gradient_Z,inequality_values,inequality_count] = add_positive_square(objective_value,gradient_Z,inequality_values,inequality_count,value_activation_moment,problem.rho_ineq,[],0,0,index_a,hidden_index,-problem.upper_s(hidden_index),0,index_a(hidden_index),1); % Penalize activation-moment violation.
    end % Finish the activation-upper-bound check.
end % Finish the hidden-unit inequality loop.
complementarity_vector = a.*(a-s); % Evaluate all nonconvex ReLU complementarity products.
complementarity_squared = sum(complementarity_vector.^2); % Compute the squared complementarity residual.
objective_value = objective_value + problem.rho_comp*complementarity_squared; % Add the genuinely nonconvex complementarity penalty.
derivative_a = 2*problem.rho_comp*complementarity_vector.*(2*a-s); % Compute the complementarity derivative with respect to a.
derivative_s = -2*problem.rho_comp*complementarity_vector.*a; % Compute the complementarity derivative with respect to s.
derivative_x = problem.W1.'*derivative_s; % Propagate the complementarity derivative to x.
derivative_z00 = problem.b1.'*derivative_s; % Propagate the complementarity derivative to z00.
gradient_Z = add_first_moment_gradient(gradient_Z,index_a,derivative_a); % Add the complementarity activation derivative.
gradient_Z = add_first_moment_gradient(gradient_Z,index_x,derivative_x); % Add the complementarity input derivative.
gradient_Z(1,1) = gradient_Z(1,1) + derivative_z00; % Add the complementarity homogenizing derivative.
gradient_Z = 0.5*(gradient_Z+gradient_Z.'); % Remove numerical asymmetry from the matrix gradient.
parts.margin = margin_value; % Store the signed margin surrogate.
parts.affine_residual = max(abs(affine_values(1:affine_count))); % Store the maximum affine equality residual.
if inequality_count > 0 % Check whether any inequality expressions were added.
    parts.inequality_residual = max(max(inequality_values(1:inequality_count),0)); % Store the maximum positive inequality violation.
else % Handle an empty inequality list.
    parts.inequality_residual = 0; % Store a zero inequality violation.
end % Finish the inequality-list check.
parts.complementarity_residual = sqrt(complementarity_squared); % Store the Euclidean complementarity residual.
parts.objective = objective_value; % Store the complete nonconvex surrogate objective.
parts.gradient_fro = norm(gradient_Z,'fro'); % Store the lifted-gradient Frobenius norm used by the direct gamma rule.
end % Finish the nonconvex lifted objective.

function gradient_Z = add_first_moment_gradient(gradient_Z,index_list,derivative_values) % Add derivatives of symmetric first-row moments to a matrix gradient.
if isempty(index_list) % Check whether there are any first moments to update.
    return; % Finish immediately for an empty index list.
end % Finish the empty-index check.
derivative_values = derivative_values(:); % Convert all derivative values to a column.
for local_index = 1:numel(index_list) % Distribute every first-moment derivative symmetrically.
    matrix_index = index_list(local_index); % Read the lifted matrix coordinate.
    gradient_Z(1,matrix_index) = gradient_Z(1,matrix_index) + 0.5*derivative_values(local_index); % Add half of the derivative to the upper-triangular entry.
    gradient_Z(matrix_index,1) = gradient_Z(matrix_index,1) + 0.5*derivative_values(local_index); % Add half of the derivative to the lower-triangular entry.
end % Finish the first-moment derivative loop.
end % Finish the symmetric first-moment gradient helper.

function [objective_value,gradient_Z,inequality_values,inequality_count] = add_positive_square(objective_value,gradient_Z,inequality_values,inequality_count,value,weight,index_x,local_x,coefficient_x,index_a,local_a,coefficient_a,coefficient_z00,diagonal_index,diagonal_coefficient) % Add one squared positive-part inequality penalty and derivative.
inequality_count = inequality_count + 1; % Increase the inequality-residual counter.
inequality_values(inequality_count) = value; % Store the raw inequality expression.
if value <= 0 % Check whether the inequality is currently satisfied.
    return; % Add no penalty or derivative for a satisfied inequality.
end % Finish the active-inequality check.
objective_value = objective_value + weight*value^2; % Add the squared positive-part penalty.
derivative_coefficient = 2*weight*value; % Compute the active penalty derivative coefficient.
if ~isempty(index_x) && local_x > 0 % Check whether the inequality depends on one input first moment.
    gradient_Z = add_first_moment_gradient(gradient_Z,index_x(local_x),derivative_coefficient*coefficient_x); % Add the input first-moment derivative.
end % Finish the input-dependence check.
if ~isempty(index_a) && local_a > 0 % Check whether the inequality depends on one activation first moment.
    gradient_Z = add_first_moment_gradient(gradient_Z,index_a(local_a),derivative_coefficient*coefficient_a); % Add the activation first-moment derivative.
end % Finish the activation-dependence check.
gradient_Z(1,1) = gradient_Z(1,1) + derivative_coefficient*coefficient_z00; % Add the homogenizing derivative.
if diagonal_index > 0 % Check whether the inequality depends on a lifted diagonal entry.
    gradient_Z(diagonal_index,diagonal_index) = gradient_Z(diagonal_index,diagonal_index) + derivative_coefficient*diagonal_coefficient; % Add the lifted diagonal derivative.
end % Finish the diagonal-dependence check.
end % Finish the scalar positive-part penalty helper.

function [objective_value,gradient_Z,inequality_values,inequality_count] = add_relu_difference_penalty(objective_value,gradient_Z,inequality_values,inequality_count,value,weight,problem,hidden_index,activation_coefficient,preactivation_coefficient) % Add the penalty for one affine ReLU difference.
inequality_count = inequality_count + 1; % Increase the inequality-residual counter.
inequality_values(inequality_count) = value; % Store the raw inequality expression.
if value <= 0 % Check whether the ReLU difference inequality is satisfied.
    return; % Add no penalty for a satisfied inequality.
end % Finish the active-inequality check.
objective_value = objective_value + weight*value^2; % Add the squared positive-part penalty.
derivative_coefficient = 2*weight*value; % Compute the active penalty derivative coefficient.
gradient_Z = add_first_moment_gradient(gradient_Z,problem.index_a(hidden_index),derivative_coefficient*activation_coefficient); % Add the activation derivative.
gradient_Z = add_first_moment_gradient(gradient_Z,problem.index_x,derivative_coefficient*preactivation_coefficient*problem.W1(hidden_index,:).'); % Add the input derivative through the preactivation.
gradient_Z(1,1) = gradient_Z(1,1) + derivative_coefficient*preactivation_coefficient*problem.b1(hidden_index); % Add the homogenizing derivative through the preactivation.
end % Finish the ReLU difference penalty helper.

function [objective_value,gradient_Z,inequality_values,inequality_count] = add_relu_upper_penalty(objective_value,gradient_Z,inequality_values,inequality_count,value,weight,problem,hidden_index,alpha_value) % Add one unstable-ReLU upper-envelope penalty.
inequality_count = inequality_count + 1; % Increase the inequality-residual counter.
inequality_values(inequality_count) = value; % Store the raw inequality expression.
if value <= 0 % Check whether the upper-envelope inequality is satisfied.
    return; % Add no penalty for a satisfied inequality.
end % Finish the active-inequality check.
objective_value = objective_value + weight*value^2; % Add the squared positive-part penalty.
derivative_coefficient = 2*weight*value; % Compute the active penalty derivative coefficient.
gradient_Z = add_first_moment_gradient(gradient_Z,problem.index_a(hidden_index),derivative_coefficient); % Add the activation derivative.
gradient_Z = add_first_moment_gradient(gradient_Z,problem.index_x,-derivative_coefficient*alpha_value*problem.W1(hidden_index,:).'); % Add the input derivative through the upper envelope.
z00_derivative = -alpha_value*problem.b1(hidden_index)+alpha_value*problem.lower_s(hidden_index); % Compute the homogenizing derivative coefficient.
gradient_Z(1,1) = gradient_Z(1,1) + derivative_coefficient*z00_derivative; % Add the homogenizing upper-envelope derivative.
end % Finish the unstable-ReLU upper-envelope helper.
