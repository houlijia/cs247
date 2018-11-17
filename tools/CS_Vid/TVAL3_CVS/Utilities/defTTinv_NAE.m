function [T,Tt] = defTTinv_NAE(ref)
% temporal difference with zero boundary
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 06/29/2010

T = @(U) FWD(U, ref);
Tt = @(U) INV(U, ref);



function V = FWD(U, ref)
V = U;
for i = 1:size(U,3)
    if i ~= ref
        V(:,:,i) = V(:,:,i) - U(:,:,ref);
    end
end
%V(:,:,ref) = U(:,:,ref)/1.3;

function V = INV(U, ref)
V = U;
%V(:,:,ref) = U(:,:,ref)*1.3;
for i = 1:size(U,3)
    if i ~= ref
        V(:,:,i) = V(:,:,i) + V(:,:,ref);
    end
end
