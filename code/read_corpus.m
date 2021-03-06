% Script for reading a corpus and convert the corpus by encoding all the
% vocabularies.
% Save the corpus with integer values and save the dictionary maps from
% words to corresponding integer value.

clear all;
fclose all;
fprintf('Reading corpus...\n');
corpus_f = fopen('../data/text8_punc');
corpus_text = textscan(corpus_f, '%s');
NN = size(corpus_text{1,1}, 1);
corpus = zeros(NN, 1);
dict = struct;
index = 1;
progress = 0;
fprintf('Building dictionary...\n');
for j = 1 : NN
    word = corpus_text{1,1}{j};
    while floor(j / NN * 80) > progress
        fprintf('.');
        progress = progress + 1;
    end
    if isfield(dict, word)
        corpus(j) = dict.(word);
    else
        dict.(word) = index;
        corpus(j) = index;
        index = index + 1;
    end 
end
fprintf('\n');
clear corpus_text;
clear corpus_text_array;
fclose(corpus_f);
save('../data/word_to_int_dict.mat', 'dict');
save('../data/corpus_int.mat', 'corpus');