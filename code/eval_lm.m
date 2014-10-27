words = input('previous words: ', 's');
words_s = strsplit(words, ' ');
N = length(words_s);
data = [dict.(char(words_s(N))), dict.(char(words_s(N))), dict.(char(words_s(N))), dict.(char(words_s(N)))];
y = ff_nn_fw(w, hyp.siz, data);
[B IX] = sort(y, 'descend');
for i = 1 : 10
    fprintf('%s:\t%f\n', char(vocabs(IX(i))), B(i));
end