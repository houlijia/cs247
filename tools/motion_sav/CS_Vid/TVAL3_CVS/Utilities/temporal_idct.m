function y = temporal_idct(x,p,q,r)
% 1D pixel-by-pixel inverse dct. Both input and output are vectors.
%
% Written by: Chengbo Li at Bell Labs

x2 = reshape(x,p*q,r)';
y2 = idct(x2)';
y = y2(:);
