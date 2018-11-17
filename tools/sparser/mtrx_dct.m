function y = mtrx_dct( x )
    % mtrx_dct - this function intends to fix an inconsistent behavior in
    % Matlab's DCT function. If x is a matrix of size (m,n), then dct(x)
    % perofms DCT of order m on each column separately. However, if m==1,
    % dct(x) performs DCT of order n on the single row!.  This function
    % performs DCT of size m on each column even if m==1 (i.e. identity).
    
    if size(x,1) == 1
        y = x;
    else
        y = dct(x);
    end
end

