function [ xcor, vec0_nrm, vec_nrms] = comp_xcor(vec0, vofst)
    %comp_xcor Compute correlation on circular measurements
    % Input
    %   vec0 - an array of order d
    %   vofst - an array of order (d,m) or a function handle such that
    %          vofst(i) returns the i-th column and vofst(0) is the number
    %          of columns
    % Output
    %   xcor - a a vector of size (m,1). The k-th entry is the dot product
    %          of msrs(:,1) and msrs(:,k+1)
    %   vec0_nrm - L2 norm of msrs(:,1)
    %   vec_nrms - the L2 norms of msrs(:2:end)
    
    if isnumeric(vofst)
        vf = @(k) vofst(:,k);
        n_col = size(vofst,2);
    else
        vf = @(k) vofst.getCol(k);
        n_col = vofst.nCols();
    end
    
    vec0_nrm = norm(vec0,2) + 1E-10;
 
    xcor = zeros(n_col,1);
    vec_nrms = xcor;
    
    for k=1:n_col
        vec1 = vf(k);
        xcor(k) = dot(vec0, vec1);
        vec_nrms(k) = norm(vec1,2) + 1E-10;
    end
    xcor = xcor ./(1E-10 + vec0_nrm*vec_nrms);
end

