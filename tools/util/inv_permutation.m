function inv_indcs = inv_permutation( indcs )
  % inv_permutation inverts a permutation
  % indcs should be a permutation of (1:n), where n=length(indcs)
  % inv_indcs is another permutation such that indcs(inv_indcs) = (1:n)
  
  n = length(indcs);
  inv_indcs = zeros(n,1);
  inv_indcs(indcs(:))  = (1:n)';
  inv_indcs = reshape(inv_indcs,size(indcs));
end

