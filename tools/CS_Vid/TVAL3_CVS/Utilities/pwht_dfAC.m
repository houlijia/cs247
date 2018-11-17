%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% define function handle A
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = pwht_dfAC(x,p,q,r,picks,perm,mode)
switch mode
    case 1
        idctx = temporal_idct(x,p,q,r);
        y = A_fWH(idctx,picks,perm);
    case 2
        dcty = At_fWH(x,picks,perm);
        y = temporal_dct(dcty,p,q,r);
    otherwise
        error('Unknown mode passed to f_handleA!');
end
