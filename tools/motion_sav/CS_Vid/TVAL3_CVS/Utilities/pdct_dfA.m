%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = pdct_dfA(x,picks,perm,mode)

% Define A*x and A'*y for a partial DCT matrix A
% Input:
%            n = interger > 0
%        picks = sub-vector of a permutation of 1:n
% Output:
%        A = struct of 2 fields
%            1) A.times: A*x
%            2) A.trans: A'*y

switch mode
    case 1
        y = pdct_n2m(x,picks,perm);
    case 2
        y = pdct_m2n(x,picks,perm);
    otherwise
        error('Unknown mode passed to f_handleA!');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = pdct_n2m(x,picks,perm)

% Calculate y = A*x,
% where A is m x n, and consists of m rows of the 
% n by n discrete-cosine transform (DCT) matrix
% with columns permuted by perm.
% The row indices are stored in picks.

tx = dct(x(perm,:));
y = tx(picks,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = pdct_m2n(y,picks,perm)

% Calculate x = A'*y,
% where A is m x n, and consists of m rows of the 
% n by n inverse discrete-cosine transform (IDCT)
% matrix with columns permuted by perm.
% The row indices are stored in picks.

N = length(perm);
[ym, yn] = size(y);
tx = zeros(N,yn);
tx(picks,:) = y;
x = zeros(N,yn);
x(perm,:) = idct(tx);
