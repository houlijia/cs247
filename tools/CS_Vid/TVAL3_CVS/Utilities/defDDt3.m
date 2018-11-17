function [D,Dt] = defDDt3
% 3D TV operators
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 06/20/2010

D = @(U) ForwardD(U);
Dt = @(X,Y,Z) Dive(X,Y,Z);


function [Dux,Duy,Duz] = ForwardD(U)

Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
Duz = cat(3, diff(U,1,3), U(:,:,1) - U(:,:,end));

function DtXY = Dive(X,Y,Z)
% DtXY = D_1' X + D_2' Y +D_3' Z

DtXY = [X(:,end,:) - X(:,1,:), -diff(X,1,2)];
DtXY = DtXY + [Y(end,:,:) - Y(1,:,:); -diff(Y,1,1)];
DtXY = DtXY + cat(3, Z(:,:,end) - Z(:,:,1), -diff(Z,1,3));
DtXY = DtXY(:);