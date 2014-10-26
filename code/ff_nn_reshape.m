function [V, D, n, H, C, U, W] = ff_nn_reshape( w, siz )
%ff_nn_reshape Roll off weight vector into weight matrices.
% Vocabulary size.
V = siz(1);

% Feature space dimension.
D = siz(2);

% Context size.
n = siz(3);

% Hidden layer size.
H = siz(4);

% Linear map from vocabulary space to feature space.
C = reshape(w(1 : V * D), V, D);

% Weights from feature space to hidden layer.
U = reshape(w(V * D + 1 : V * D + (D * n + 1) * H), (D * n + 1), H);

% Weights from hidden layer to output layer.
W = reshape(w(V * D + (D * n + 1) * H + 1 : V * D + (D * n + 1) * H + (H + 1) * V), H + 1, V);
end