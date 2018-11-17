function [A1,A2,b,Y,TY] = comp_3D_fbf(cellY,p,q,r,ratio,ref,scl)
% Compress 3D video cube using compressive sensing, meanwhile generate
% function handle A as the measurement matrix 
%
% cellY: the cell which stores one of Y U V
% p,q,r: dimensions
% ratio: compression ratio
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 07/10/2010

[T, Tinv] = defTTinv_NAE(ref);
Y = zeros(p,q,r);
for i = 1:r
    Y(:,:,i) = cellY{i};
end

N = p*q;
M1 = round(ratio(1)*N);
M2 = round(ratio(2)*N);

% truncate the temporal differences
TY = T(Y);
TY2 = abs(TY).^2/scl;

% generate measurement matrix
p = randperm(N);
picks1 = p(1:M1);
picks1(1) = 1;
for ii = 2:M1
    if picks1(ii) == 1
        picks1(ii) = p(M1+1);
        break;
    end
end
perm = randperm(N); % column permutations allowable
A1 = @(x,mode) pwht_dfA(x,picks1,perm,mode);

picks2 = p(1:M2);
picks2(1) = 1;
for ii = 2:M2
    if picks2(ii) == 1
        picks2(ii) = p(M2+1);
        break;
    end
end
A2 = @(x,mode) pwht_dfA(x,picks2,perm,mode);

% observation
b = cell(1,r);
for i = 1:r
    if i ~= ref     b{i} = A1(TY2((i-1)*N+1:i*N)',1);   end
end
b{ref} = A2(TY2((ref-1)*N+1:ref*N)',1);

