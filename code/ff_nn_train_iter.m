function [ w, dw, CE ] = ff_nn_train_iter( w, dw, data, target, hyp)
    [~, cost_deriv, CE] = ff_nn_cost(w, hyp.siz, data, target, hyp.reg);
    dw_2 = -hyp.lr * cost_deriv;
    w = w + dw_2 + hyp.mom * dw;
    dw = dw_2;
end