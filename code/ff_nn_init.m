hyp = struct;
hyp.init = 0.001;
hyp.lr = 0.01;
hyp.mom = 0.1;
hyp.reg = 5e-5;
hyp.epochs = 10;
hyp.epoch_siz = 1000;
hyp.epoch_vsiz = 100;
hyp.ctx = 5;
V = length(vocabs);
hyp.siz = [V; 100; hyp.ctx - 1; 10];
L = hyp.siz(1) * hyp.siz(2) + ...
    (hyp.siz(2)*hyp.siz(3)+1) * hyp.siz(4) + ...
    (hyp.siz(4)+1) * hyp.siz(1);
rng('shuffle', 'twister');
w = rand(L,1) * hyp.init;