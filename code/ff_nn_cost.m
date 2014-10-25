function [ cost, cost_deriv ] = ff_nn_cost( w, siz, data, target )
%ff_nn cost Compute the cost function and weight derivatives for a
%one hidden layer feedforward neural net language model.
% ff_nn_cost( weights, examples )
% Input:
%       w (Lx1 vector): Weights of the neural nets, flattened. Let D be the
%       input size, H be the hidden layer size, K be the output size. The
%       first (D+1)xH is the weights from input layer to hidden layer. The
%       second (H+1)xK is the weights from hidden layer to output layer.
%       The one more weight functions as bias weight.
%
%       siz (3x1 vector): Size of each layer, input layer, hidden layer and
%       output layer, respectively.
%       
%       data (NxD vector): Training data. N is the size of a batch. D is 
%       input dimension.
%
%       target (NxK vector): Training target. N is the size of a batch.
%
% Output:
%       cost (scalar): Value of cost function (cross entropy).
%       cost_deriv (Lx1 vector): Cost function derivative with regard to
%       weights.
%
% Created by: Mengye Ren
% Date: 24-OCT-2014


end
