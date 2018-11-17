function yuv = vecRGB2YUV(rgb, pxmx)
  % Convert RGB values into YUV values. The exact conversion is from RGB
  % to YCbCr for HDTV, scaled to the full range of 0. Based on 
  % http://www.equasys.de/colorconversion.html
  % Input:
  %   rgb - can be a Nx3 array containing RGB values, or a vector which is
  %        reshaped into a Nx3 vector, or a cell array of 3 entries
  %        containing the Y,U,V compoents.
  %   pxmx - (optional) pixel max. Default: 255
  % Output
  %   rgb values in Nx3 representation, or if yuv is a non-vector multi-dim
  %   array, the original size of yuv.
  
  if nargin < 2
    pxmx = 255;
  end
  
  if iscell(rgb)
    r = zeros(numel(rgb{1}),3);
    for k=1:3
      r(:,k) = rgb{k}(:);
    end
    rgb = r;
    sz = size(rgb);
  elseif isvector(rgb)
    rgb = reshape(rgb, [length(rgb)/3, 3]);
    sz = size(rgb);
  else
    sz = size(rgb);
    rgb  = reshape(rgb, [numel(rgb)/3, 3]);
  end
  
  scl_fctr = single(255/pxmx);
  rgb = rgb*scl_fctr;
  
  % Conversion coefficients from RGB to YCbCr for HDTV, scaled to the
  % full available range. Based on http://www.equasys.de/colorconversion.html
  cnvrt_mtrx = single(diag(255./[219,224,224]) * [...
    0.183, 0.614, 0.062;...
    -0.101, -0.339, 0.439; ...
    0.439, -0.399, -0.040]);
  cnvrt_ofst = single([0;128;128]) * ones(1,size(rgb,1),'single');
  
  yuv = cnvrt_mtrx * rgb' + cnvrt_ofst;
  yuv = yuv' / scl_fctr;
  yuv = reshape(yuv,sz);
end
  
  
