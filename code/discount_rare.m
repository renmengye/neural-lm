clear all;
consts;
corpus = load(CORPUS_INT);
corpus = corpus.corpus;
dict_old = load(DICT);
dict_old = dict_old.dict;
L = length(corpus);
vocabs = fieldnames(dict_old);
N = length(vocabs);
edges = 1 : N;

% One-gram stats.
one_gram_stats = histc(corpus, edges);
CUTOFF = 5;
dict = struct;
newidx = 1;

% Assign new id in the dict.
fprintf('Building new dict...\n');
progress = 0;
for i = 1 : N
    while floor(i / N * 80) > progress
        fprintf('.');
        progress = progress + 1;
    end
    if one_gram_stats(i) <= CUTOFF
        fprintf('%s\n', char(vocabs(i)));
    else
        dict.(char(vocabs(i))) = newidx;
        newidx = newidx + 1;
    end
end
dict.unk_ = newidx;
fprintf('\n');

% Assign new id in the corpus.
fprintf('Building new corpus...\n');
progress = 0;
for i = 1 : L
    while floor(i / L * 80) > progress
        fprintf('.');
        progress = progress + 1;
    end
    word = char(vocabs(corpus(i)));
    if isfield(dict, word)
        corpus(i) = dict.(word);
    else
        corpus(i) = dict.unk_;
    end
end
fprintf('\n');

save(CORPUS_INT_SHORT, 'corpus');
save(DICT_SHORT, 'dict');