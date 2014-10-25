% Script for dividing the corpus into training, validation, and test set.
clear all;
consts;
fclose all;
sent_mark = 'period_';

% Constants.
train_ratio = 0.75;
kfold = 10;
valid_ratio = train_ratio / kfold;
test_ratio = 1 - train_ratio;

% Read Nx1 vector where N is the length of the corpus (in words).
corpus = load(CORPUS_INT_SHORT);
corpus = corpus.corpus;
dict = load(DICT_SHORT);
dict = dict.dict;
N = length(corpus);
valid_size = floor(N * valid_ratio);
test_size = floor(N * test_ratio);
sent = dict.(sent_mark);

% Results.
train_data = zeros(kfold * valid_size, 1);
valid_data = zeros(valid_size, 1);
test_data = corpus(N - test_size + 1 : N);
save(TEST_DATA, 'test_data');

for k = 1 : kfold
    if (k == 1)
        train_data = corpus(valid_size + 1 : N);
        valid_data = corpus(1 : valid_size);
    elseif (k == kfold)
        train_data = corpus(1 : N - valid_size);
        valid_data = corpus(N - valid_size + 1 : N);
    else
        train_data(1 : k * valid_size) = corpus(1 : k * valid_size);
        valid_data = corpus(k * valid_size + 1 : (k+1) * valid_size);
        train_data(k * valid_size + 1 : end) = corpus((k+1) * valid_size + 1 : N);
    end
    save(sprintf(TRAIN_DATA, k-1), 'train_data');
    save(sprintf(VALID_DATA, k-1), 'valid_data');
end
