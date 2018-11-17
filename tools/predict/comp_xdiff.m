function [ xcor,vec0_nrm] = comp_xdiff(vec0, vofst, nrm_exp)
    %comp_xdiff Compute differences on circular measurements
    % Input
    %   vec0 - an array of order d
    %   vofst - an array of order (d,m) or a function handle such that
    %          vofst(i) returns the i-th column and vofst(0) is the number
    %          of columns
    %   nrm_exp - (optional, default=1) norm exponent. 
    % Output
    %   xcor - a vector of size (m,1). The k-th entry is the L1 norm of
    %          difference between msrs(:,1) and msrs(:,k+1)
    %   vec0_nrm - L1 norm of msrs(:,1)/vec0_nrm
    
    if nargin < 2
        nrm_exp = 1;
    end
    
    if isnumeric(vofst)
        vf = @(k) vofst(:,k);
        n_col = size(vofst,2);
    else
        vf = @(k) vofst.getCol(k);
        n_col = vofst.nCols();
    end
    
    vec0_nrm = norm(vec0,nrm_exp);
    
    xcor = zeros(n_col,1);
    for k=1:n_col
        vec1 = vf(k);
        xcor(k) = norm(vec0-vec1,nrm_exp);
    end
    xcor = 1 - xcor/(vec0_nrm + 1E-10);
    
end
