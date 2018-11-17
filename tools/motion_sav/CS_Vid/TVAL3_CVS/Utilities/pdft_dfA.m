%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = pdft_dfA(x,picks,perm,mode)

% Define A*x and A'*y for a partial DFT matrix A
% Input:
%            n = interger > 0
%        picks = sub-vector of a permutation of 1:n
% Output:
%        A = struct of 2 fields
%            1) A.times: A*x
%            2) A.trans: A'*y

switch mode
    case 1
        y = pdft_n2m(x,picks,perm);
    case 2
        y = pdft_m2n(x,picks,perm);
    otherwise
        error('Unknown mode passed to f_handleA!');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = pdft_n2m(x,picks,perm)

% Calculate y = A*x,
% where A is m x n, and consists of m rows of the 
% n by n discrete-Fourier transform (FFT) matrix.
% The row indices are stored in picks.

x = x(:);
n = length(x);
tx = fft(x(perm))/sqrt(n);
y = tx(picks);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = pdft_m2n(y,picks,perm)

% Calculate x = A'*y,
% where A is m x n, and consists of m rows of the 
% n by n inverse discrete-Fourier transform (IFFT) 
% matrix. The row indices are stored in picks.

n = length(perm); 
tx = zeros(n,1);
tx(picks) = y;
x = zeros(n,1);
x(perm) = ifft(tx)*sqrt(n);
