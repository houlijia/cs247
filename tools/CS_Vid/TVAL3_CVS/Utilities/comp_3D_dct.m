function [A,b,Y,dctY] = comp_3D_dct(cellY,p,q,r,scl,ratio,ndb,dct_bar,opt)
% Compress 3D video cube using compressive sensing, meanwhile generate
% function handle A as the measurement matrix 
%
% cellY  : the cell which stores one of Y U V
% p,q,r  : dimensions
% ratio  : compression ratio
% ndb    : channel noise (decibel)
% dct_bar: threshold for large coeffs scaling in each DCT frame except the 1st one
% opt    : choose between 'dct' and 'wht'
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 07/15/2010

Y = zeros(p,q,r);
for i = 1:r
    Y(:,:,i) = cellY{i};
end

dctY = reshape(temporal_dct(Y(:),p,q,r),p,q,r)/scl;

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
    % display('Measurement matrix: partial DCT with permutations.');
else
    A = @(x,mode) pwht_dfA(x,picks,perm,mode);
    % display('Measurement matrix: partial Walsh Hadamard with permutations.');
end

% scale those large coefficients in each DCT frame except the 1st one
dctY2 = dctY;
if exist('dct_bar')==1
    peak = max(abs(dctY(:)));
    for i = 2:r
        subpk(i) = max(max(abs(dctY(:,:,i))));
        idx = find(abs(dctY(:,:,i)) > subpk(i)/dct_bar);    % amplify more coeffs for higher noise level
        dctY2((i-1)*slx*sly+idx) = dctY((i-1)*slx*sly+idx)*(peak/subpk(i));
    end
end
        
% observation
b = A(dctY2(:),1);

% add noise
if exist('ndb')==1
noise = randn(N,1);          % length depends on bandwith
sigma = norm(b(2:end))/norm(noise(2:end))/abs(10^(ndb/20));     % neglect the first coeff
b = b + sigma*noise(1:M);
end
