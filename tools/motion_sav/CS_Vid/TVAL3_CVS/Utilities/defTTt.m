function [T,Tt] = defTTt
% temporal difference with zero boundary

T = @(U) cat(3, diff(U,1,3), - U(:,:,end));
Tt = @(Z) cat(3, - Z(:,:,1), -diff(Z,1,3));