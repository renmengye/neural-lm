function [ w ] = ff_nn_train( w, siz, train_data, train_target);
    epochs = 3000;
    costs = zeros(epochs, 1);
    [N, ~] = size(train_data);
    for i = 1 : epochs
        j = 1;    
        %j = ceil(rand(1,1) * N);
            %train_data(j, :)
            %train_target(j)
            [cost, cost_deriv] = ff_nn_cost(w, siz, train_data(j, :), train_target(j), 0.1);
            costs(i) = costs(i) + cost;
            w = w - 0.01 * cost_deriv;
        
        fprintf('Epoch: %d, CE: %f\n', i, costs(i));
    end
    plot(costs);
end