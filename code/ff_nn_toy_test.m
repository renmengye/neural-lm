clear all;
data = [1 2 3; 3 1 2; 2 3 1];
target = [1 2 3];
siz = [3 3 3 3];

hyp = struct;
hyp.init = 1;
hyp.lr = 1;
hyp.mom = 0.01;
hyp.reg = 3e-4;
hyp.epochs = 10;

figure;
hold on;

%lr = [0.001, 0.01, 0.1, 0.3, 1];
%mom = [0.001, 0.01, 0.1, 0.3, 1];
%reg = [0.0001, 0.0003, 0.001, 0.003, 0.01];
cmap = hsv(5);
markers = [{'-+'}, {'-o'}, {'-*'}, {'-x'}, {'-s'}, {'-d'}, {'-p'}];
%for i = 1:5
%rng(1);
%rng('shuffle', 'twister');
w = rand(51,1) * hyp.init;
dw = 0;
for i = 1 : hyp.epochs
    [w, dw, cost] = ff_nn_train_iter(w, dw, siz, data, target, hyp);
    fprintf('Epoch: %d, CE: %f\n', i, cost);
end
ff_nn_fw(w,siz,[1 2 3; 3 1 2; 2 3 1])
%plot(costs, char(markers(mod(i, 7) + 1)), 'Color', cmap(i, :));
%end
%legend('0.0001', '0.0003', '0.001', '0.003', '0.01', 'Location', 'Best');