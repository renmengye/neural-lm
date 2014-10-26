function [ data, target ] = get_ctx_window( corpus, idx, n )
%get_ctx_window Get a context window of the corpus.
%
% Input:
%       corpus (Nx1 vector): vector of numbers (encoded from a corpus).
%       idx (scalar): the zero-based index of the window.
%       n (scalar): context window size.
%
% Output:
%       data ((n-1)x1 vector): previous (n-1) words.
%       target (scalar): the last word.
%

% Parameter check.
% N = length(corpus);
% if n < 2
%     error('Context window is too small');
% end
% if( idx + n - 1 > N )
%     error('Exeed corpus length');
% end
N = length(corpus);
idx = mod(idx, N - n + 1);
p = idx + 1;
q = idx + n;
data = corpus(p : q - 1)';
target = corpus(q);

end