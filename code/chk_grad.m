function [ diff, deriv, deriv_chk ] = chk_grad( w, siz, data, target, wt_reg, cost_func);
[L, ~] = size(w);
[~, dE_dW] = cost_func( w, siz, data, target, wt_reg );
dE_dW_chk = zeros(size(dE_dW));
eps = 1e-2;
for i = 1 : L
    w_temp = w;
    w_temp(i) = w(i) + eps;
    [E_temp_1, ~] = cost_func(w_temp, siz, data, target, wt_reg);
    w_temp(i) = w(i) - eps;
    [E_temp_2, ~] = cost_func(w_temp, siz, data, target, wt_reg);
    dE_dW_chk(i) = (E_temp_1 - E_temp_2) / (2*eps);
end
deriv = dE_dW;
deriv_chk = dE_dW_chk;
diff = dE_dW_chk - dE_dW;
end