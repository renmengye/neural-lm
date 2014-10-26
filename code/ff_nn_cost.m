function [ cost, cost_deriv, CE ] = ff_nn_cost( w, siz, data, target, wt_reg )
%ff_nn_cost Compute the cost function and weight derivatives for a
%one hidden layer feedforward neural net language model.
% ff_nn_cost( weights, examples )
% Input:
%       w (Lx1 vector): Weights of the neural nets, flattened. Let V be the
%       vocabulary size, D be the feature vector size, n be the context size,
%       H be the hidden layer size. The first VxD is the weights from 
%       vocabulary one-hot vector to feature vector. The second (Dn+1)xH 
%       is the weights from input layer to hidden layer. The second (H+1)xV
%       is the weights from hidden layer to output layer. The one more 
%       weight functions as bias weight.
%
%       siz (4x1 vector): Size of each layer, input layer (V = vocabulary 
%       size), feature space layer (D = feature dimension), hidden layer 
%       (H = hidden layer size), and output layer (V = vocabulary size), 
%       respectively.
%       
%       data (Nxn vector): Training data. N is the size of a batch. Each
%       element is the class index of the input vocanulary of previous n 
%       words.
%
%       target (Nx1 vector): Training target. N is the size of a batch.
%       Each element is the class index of the target.
%
%       wt_reg (scalar): weight regularization constant.
%
% Output:
%       cost (scalar): Value of cost function (cross entropy).
%       cost_deriv (Lx1 vector): Cost function derivative with regard to
%       weights.
%
% Created by: Mengye Ren
% Date: 24-OCT-2014

[V, D, n, H, C, U, W] = ff_nn_reshape(w, siz);
[y, h, f] = ff_nn_fw(w, siz, data);
[N, ~] = size(data);
E = 0;
for ex_i = 1 : N
    E = E - log(y(ex_i, target(ex_i)));
end
dE_dz = y;
for ex_i = 1 : N
    dE_dz(ex_i, target(ex_i)) = y(ex_i, target(ex_i)) - 1;
end
dE_dW = h' * dE_dz;
dE_dh = dE_dz * W(2:end, :)';
dh_dq = (1 - h(:,2:end)) .* (1 + h(:,2:end));
dE_dq = dE_dh .* dh_dq;
dE_dU = f' * dE_dq;
dE_dC = zeros(V, D);

for ex_i = 1 : N
    dE_df = dE_dq(ex_i, :) * U(2:end, :)';
    dE_df = reshape(dE_df, [n, D]);
    x = zeros(V, n);
    for ctx_i = 1 : n
        x(data(ex_i, ctx_i), ctx_i) = 1;
    end
    dE_dC = dE_dC + x * dE_df;
end
C_ = reshape(C, [V*D, 1]);
U_ = U;
U_(1,:) = 0;
U_ = reshape(U_, [(D * n + 1) * H, 1]);
W_ = W;
W_(1,:) = 0;
W_ = reshape(W_, [(H + 1) * V, 1]);
cost = E + wt_reg * ((C_' * C_) + (U_' * U_) + (W_' * W_));
CE = E;
cost_deriv = [
    reshape(dE_dC, [V * D, 1]);...
    reshape(dE_dU, [(D * n + 1) * H, 1]);...
    reshape(dE_dW, [(H + 1) * V, 1])
    ];
cost_deriv = cost_deriv + 2 * wt_reg * [C_; U_; W_];

end