function y=ifwh_hadamard(y)
  % Simple implementation of inverse Fast Walsh-Hadamard transform in 
  % Hadamard order (Kronecker product of [1 1; 1 -1]).
  % Input:
  %   y - a numeric array of any order. The transform is performed over the
  %       first dimension. If the first dimension is not a power of 2 zero
  %       padding is done.
  % Output:
  %   y - Output. Same class and size as input, except that the first
  %        dimension is extended to next power of 2.
  
  if isempty(y)
    return
  end
  
  sz = size(y);
  N = 2^nextpow2(sz(1));
    
  y = cat(1, y, zeros([(N-sz(1)) sz(2:end)], 'like', y));
  sz = size(y);
  L = numel(y);
  M=1;
  r_p = (1:2:L);
  if isa(y,'gpuArray')
    r_p = gpuArray(r_p);
  end
  r_n = r_p+1;
    
  while (M<N)
    L2 = L/2;
    y = reshape(y, M,L);
    b_p = y(:,r_p(1:L2));
    b_n = y(:,r_n(1:L2));
    y(:,r_p(1:L2)) = b_p + b_n;
    y(:,r_n(1:L2)) = b_p - b_n;
    M = M*2;
    L = L2;
  end
  
  y = reshape(y,sz);
  
end