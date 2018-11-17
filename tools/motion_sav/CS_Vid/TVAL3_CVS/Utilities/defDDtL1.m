function [D,Dt] = defDDtL1
% TV on frame 1 + L1 on the other frames
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 07/02/2010

D = @(U) ForwardD(U);
Dt = @(X,Y) Dive(X,Y);


function [Dux,Duy] = ForwardD(U)

Dux = U/3;
Duy = zeros(size(U));
frm = U(:,:,1);
Dux(:,:,1) = [diff(frm,1,2), frm(:,1) - frm(:,end)];
Duy(:,:,1) = [diff(frm,1,1); frm(1,:) - frm(end,:)];

function DtXY = Dive(X,Y)
% DtXY = D_1' X + D_2' Y

DtXY = X*3;
frmX = X(:,:,1);
frmY = Y(:,:,1);
DtXY(:,:,1) = [frmX(:,end) - frmX(:,1), -diff(frmX,1,2)];
DtXY(:,:,1) = DtXY(:,:,1) + [frmY(end,:) - frmY(1,:); -diff(frmY,1,1)];
% DtXY = DtXY(:);