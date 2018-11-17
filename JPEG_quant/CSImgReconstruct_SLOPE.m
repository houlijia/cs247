function [img, raw_img] = CSImgReconstruct_SLOPE(y, para, A, At )
  % CSImgReconstruct_SLOPE reconstructs an image from measurements using SLOPE
  % 
  % Input:
  %   y: measurements vector
  %   para: a struct containing parameters. Required fields:
  %           row: number of rows in the image
  %           col: number of columns in the image
  %           CSr: Compression ratio
  %         optional fields (with defaults)
  %           lambda: [2]
  %           iter: number of iterations [100]
  %           patch: [8]
  %           step: [1]
  %           cluster: [200]
  %           T_s: Space transformation used ['dct']
  %           T_t: Third dimension transformation ['wavelent']
  %           d3: If true use 3d transformation [1]
  %           ratio3d: [0.6]
  %   A: A function handle such that A(x) is the sensing matrix times x
  %   At: A function handle such that A(x) is the transposed sensing matrix
  %       times x, divided by the square of the norm  of the matrix.
  % Output:
  %   img: The normalized image
  %   raw_img: The output of the reconstruction algorithm, before, normalization
  %           
  
  if ~isfield(para, 'lambda')
    para.lambda = 2;
  end
  if ~isfield(para, 'iter')
    para.iter = 100;
  end
  if ~isfield(para, 'patch')
    para.patch = 8;
  end
  if ~isfield(para, 'step')
    para.step = 1;
  end
  if ~isfield(para, 'cluster')
    para.cluster = 100;
  end
  if ~isfield(para, 'T_s')
    para.T_s = 'dct';
  end
  if ~isfield(para, 'T_t')
    para.T_t = 'wavelet';
  end
  if ~isfield(para, 'd3')
    para.d3 = 1;
  end
  if ~isfield(para, 'ratio3d')
    para.ratio3d = 0.6;
  end
  
  y = gather(y);  % prevent GPU processing
  raw_img = GAP_SLOPE_3D(y, para, A, At );
  img = raw_img;
end

