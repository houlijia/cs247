function [T,Tt] = defTTt_NAE
% temporal difference with zero boundary
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 06/28/2010

T = @(U) FWD(U);
Tt = @(U) BWD(U);



function V = FWD(U)
V = U;
for i = 1:(size(U,3)-1)
    V(:,:,i) = V(:,:,i) - U(:,:,end);
end

function V = BWD(U)
V = U;
for i = 2:size(U,3)
    V(:,:,i) = V(:,:,i) - U(:,:,1);
end
