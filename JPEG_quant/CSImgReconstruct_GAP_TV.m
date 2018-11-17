function [img, raw_img] = CSImgReconstruct_GAP_TV(y, para, A, At )
  % CSImgReconstruct_GAP_TV reconstructs an image from measurements using GAP_TV
  % 
  % Input:
  %   y: measurements vector
  %   para: a struct containing parameters. Required fields:
  %           row: number of rows in the image
  %           col: number of columns in the image
  %         optional fields (with defaults)
  %           iter: number of iterations [200]
  %           lambda: [1]
  %           TVweight: [0.07]
  %   A: A function handle such that A(x) is the sensing matrix times x
  %   At: A function handle such that A(x) is the transposed sensing matrix
  %       times x, divided by the square of the norm  of the matrix.
  % Output:
  %   img: The normalized image
  %   raw_img: The output of the reconstruction algorithm, before, normalization
  %           
  
  if ~isfield(para, 'iter')
    para.iter = 100;
  end
  if ~isfield(para, 'lambda')
    para.lambda = 1;
  end
  if ~isfield(para, 'tv_wgt')
    para.tv_wgt = 0.07;
  end
  
  raw_img = TV_GAP(y, 1, para, A, At);
  img = raw_img/max(raw_img(:));
  raw_img = uint8(round(raw_img*255));
end

