function [ y, h, f ] = ff_nn_fw( w, siz, data )
%ff_nn_fw Forward propagate the neural nets. Compute the probability of the
%current word given previous n words.
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
%       size), feature space layer (D = feature dimension), context size 
%       (n = number of previous words seen), hidden layer (H = hidden layer
%       size). Output layer size is the same as input (V).
%       
%       data (Nxn vector): Input data. N is the size of a batch. Each
%       element is the class index of the input vocanulary of previous n 
%       words.
%
% Output:
%       y (Vx1 vector): Each element is the probability of the word v given
%       the previous n words.
%
% Created by: Mengye Ren
% Date: 24-OCT-2014

[V, D, n, H, C, U, W] = ff_nn_reshape(w, siz);
[N, ~] = size(data);
y = zeros(N, V);
h = zeros(N, H + 1);
f = zeros(N, D * n + 1);
for i = 1 : N
    % Feature vector of n words.
    f_temp = C(data(i, :), :);
    f(i, :) = [0 reshape(f_temp, [1, D * n])];

    % Hidden units.
    h(i, :) = [0 tansig(f(i, :) * U)];

    % Output.
    y(i, :) = softmax((h(i, :) * W)');
end
end