function [A,b,Y] = comp_3D(cellY,p,q,r,ratio,sigma,opt)
% Compress 3D video cube using compressive sensing, meanwhile generate
% function handle A as the measurement matrix 
%
% cellY: the cell which stores one of Y U V
% p,q,r: dimensions
% ratio: compression ratio
% sigma: noise level (optional)
% opt  : choose between 'dct' and 'wht'
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 07/14/2010

Y = zeros(p,q,r);
for i = 1:r
    Y(:,:,i) = cellY{i};
end

N = p*q*r;
M = round(ratio*N);

% generate measurement matrix
p = randperm(N);
picks = p(1:M);
picks(1) = 1;
for ii = 2:M
    if picks(ii) == 1
        picks(ii) = p(M+1);
        break;
    end
end
perm = randperm(N); % column permutations allowable

% generate function handle A
if ~exist('opt')
    opt = 'wht';
end
if opt == 'dct'
    A = @(x,mode) pdct_dfA(x,picks,perm,mode);
    display('Measurement matrix: partial DCT with permutations.');
else
    A = @(x,mode) pwht_dfA(x,picks,perm,mode);
    % display('Measurement matrix: partial Walsh Hadamard with permutations.');
end

% observation
b = A(Y(:),1);

% add noise
if exist('sigma')==1
    bavg = mean(abs(b));
    noise = randn(M,1);
    b = b + sigma*bavg*noise;
end
