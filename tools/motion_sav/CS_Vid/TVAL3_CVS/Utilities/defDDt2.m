function [D,Dt] = defDDt2
% spatial difference with periodic boundary
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 06/20/2010

D = @(U) ForwardD(U);
Dt = @(X,Y) Dive(X,Y);


function [Dux,Duy] = ForwardD(U)

Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];

function DtXY = Dive(X,Y)
% DtXY = D_1' X + D_2' Y

DtXY = [X(:,end,:) - X(:,1,:), -diff(X,1,2)];
DtXY = DtXY + [Y(end,:,:) - Y(1,:,:); -diff(Y,1,1)];
% DtXY = DtXY(:);