function y = fwh_hadamard(y)
  % Simple implementation of fast Walsh-Hadamard transform in 
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
  
  y = ifwh_hadamard(y);
  y = y / size(y,1);
  
end

