function rgb = vecYUV2RGB(yuv, pxmx)
  % Convert YUV values into RGB values. The exact conversion is from YCbCr
  % to RGB for HDTV, where the YCbCr is scaled to the full range of 0. 
  % Based on   % http://www.equasys.de/colorconversion.html
  % Input:
  %   yuv - can be a Nx3 array containing RGB values, or a vector which is
  %        reshaped into a Nx3 vector, or a cell array of 3 entries
  %        containing the Y,U,V compoents.
  %   pxmx - (optional) pixel max. Default: 255
  % Output
  %   rgb values in Nx3 representation

  if nargin < 2
    pxmx = 255;
  end
  
  if iscell(yuv)
    y = zeros(numel(yuv{1}),3);
    for k=1:3
      y(:,k) = yuv{k}(:);
    end
    yuv = y;
    sz = size(yuv);
  elseif isvector(yuv)
    yuv = reshape(yuv, [length(yuv)/3, 3]);
    sz = size(yuv);
  else
    sz = size(yuv);
    yuv  = reshape(yuv, [numel(yuv)/3, 3]);
  end
  
  scl_fctr = single(255/pxmx);
  yuv = yuv*scl_fctr;
  
  % Conversion coefficients from RGB to YCbCr for HDTV, scaled to the
  % full available range. Based on http://www.equasys.de/colorconversion.html
  cnvrt_mtrx = single([...
    1.164, 0, 1.793;...
    1.164, -0.213, -0.533; ...
    1.164, 2.112, 0] * diag([219,224,224]/255));
  cnvrt_ofst = single([0;128;128]) * ones(1,size(yuv,1),'single');
  
  rgb = cnvrt_mtrx *(yuv' - cnvrt_ofst);
  rgb = rgb' / scl_fctr;
  rgb = reshape(rgb,sz);
end
  
