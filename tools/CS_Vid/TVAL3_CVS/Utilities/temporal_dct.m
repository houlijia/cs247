function y = temporal_dct(x,p,q,r)
% 1D pixel-by-pixel dct. Both input and output are vectors.
%
% Written by: Chengbo Li at Bell Labs

x2 = reshape(x,p*q,r)';
y2 = dct(x2)';
y = y2(:);
