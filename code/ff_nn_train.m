N = length(train_data) - hyp.ctx;
Nvalid = length(valid_data) - hyp.ctx;
CE = zeros(hyp.epochs, hyp.epoch_siz);
CE_v = zeros(hyp.epochs, hyp.epoch_vsiz);
timestr = datestr(now, 'yyyy-mm-dd-HH-MM-ss');
for ep_i = 1 : hyp.epochs
    dw = 0;
    rng('shuffle', 'twister');
    % Training, pick some random context window from the train corpus.
    for ctx_i = 0 : hyp.epoch_siz - 1
        ctx_k = ceil(rand(1,1) * N);
        [data, target] = get_ctx_window(train_data, ctx_k, hyp.ctx);
        [w, dw, ce] = ff_nn_train_iter(w, dw, data, target, hyp);
        CE(ep_i, ctx_i + 1) = ce;
        if mod(ctx_i + 1,100) == 0
            fprintf('Epoch: %d, Train CE: %f\n', ep_i, mean(CE(ep_i, 1:ctx_i+1), 2));
        end
    end
    
    % Validation, pick some random context windows from the valid corpus.
    for ctx_i = 0 : hyp.epoch_vsiz - 1
        ctx_k = ceil(rand(1,1) * Nvalid);
        [data, target] = get_ctx_window(train_data, ctx_k, hyp.ctx);
        [~, ~, ce] = ff_nn_cost(w, hyp.siz, data, target, hyp.reg);
        CE_v(ep_i, ctx_i + 1) = ce;
    end
    fprintf('Epoch: %d, Valid CE: %f\n', ep_i, mean(CE_v(ep_i, :), 2));
    hyp.lr = hyp.lr * hyp.lr_decay;
    save(strcat('../lm/ff-', timestr, '.mat'), 'w', 'hyp');
end