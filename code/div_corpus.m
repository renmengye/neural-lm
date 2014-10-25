% Script for dividing the corpus into training, validation, and test set.
sent_mark = 'period_';
clear all;
fclose all;
fprintf('Reading int corpus...');

% Read Nx1 vector where N is the length of the corpus (in words).
corpus = load('corpus_int');
N = length(corpus);
