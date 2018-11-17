function [mu,b,scl] = Scaleb(mu,b,option)

% Scales mu and f so that the finite difference of f is neither too small 
% nor too large.
%
% If option is assigned, mu will be scaled accordingly.
%
% Written by: Chengbo Li



scl = 1/256;  % for 3D total variation minimization
%scl = 1/10;

b = scl*b;

if option
    mu = mu/scl;
end

return