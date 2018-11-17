function [y,prmt] = GetMeasurements(I, CSr, do_permut)
  % Compute measurements of a vector using Walsh Hadamard matrix
  % INPUT:
  %   I - measurements vector
  %   CSr - compression ratio
  %   do_permut - it true, permute input vector before muliplying by WH matrix
  % OUTPUT:
  %   y - measurements vector
  %   prmt - If do_permut is true, this is the permutation vector. Otherwise
  %          empty.
if nargin < 3
  do_permut = flase;
end
I = double(I);
[row,col,~] = size(I);
Mea_num = round(CSr*row*col);
Phi_index = Hadamard_index_zigzag(row*col,Mea_num);
A = @(z) A_wht_ord(z, Phi_index);
%At = @(z) At_wht_ord(z, Phi_index, row, col);
if do_permut
  prmt = randperm(length(I(:)));
  I = I(prmt);
else
  prmt = [];
end
y = A(I(:));%para.lambda = 2e-6; 


end