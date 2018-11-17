function y = mtrx_idct( x )
    % mtrx_dct - this function intends to fix an inconsistent behavior in
    % Matlab's IDCT function. If x is a matrix of size (m,n), then idct(x)
    % perofms IDCT of order m on each column separately. However, if m==1,
    % idct(x) performs IDCT of order n on the single row!.  This function
    % performs IDCT of size m on each column even if m==1 (i.e. identity).
    
    if size(x,1) == 1
        y = x;
    else
        y = idct(x);
    end
end


