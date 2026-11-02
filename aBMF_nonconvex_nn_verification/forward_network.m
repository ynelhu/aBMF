function score = forward_network(dataset,x) % Evaluate the fixed one-hidden-layer binary ReLU classifier.
preactivation = dataset.W1*x + dataset.b1; % Compute all hidden-layer preactivations.
activation = max(preactivation,0); % Apply the ReLU activation elementwise.
score = dataset.w2.'*activation + dataset.b2; % Compute the scalar signed class score.
end % Finish the network forward pass.
