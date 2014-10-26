data = [1:3; 2:4; 7:9];
target = (1:3)';
siz = [10, 5, 3, 5];
L = siz(1) * siz(2) + (siz(2)*siz(3)+1) * siz(4) + (siz(4)+1) * siz(1);
w = rand(L, 1) * 0.001;
[ diff, deriv, deriv_chk ] =  chk_grad(w, siz, data,target, 0.1, @ff_nn_cost);
if sum(diff < 1e-5) == L
    fprintf('Gradient check success.\n');
else
    error('Gradient check failed.');
end