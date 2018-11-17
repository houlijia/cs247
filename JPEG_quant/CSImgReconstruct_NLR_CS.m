function [img, raw_img] = CSImgReconstruct_NLR_CS(y, para, A, At )
  % CSImgReconstruct_NLR_CS reconstructs an image from measurements using NLR_CS
  % 
  % Input:
  %   y: measurements vector
  %   para: a struct containing parameters. Required fields:
  %           row: number of rows in the image
  %           col: number of columns in the image
  %           CSr: Compression ratio
  %         optional fields (with defaults)
  %           s_model: [1]
  %           L_threshold: [20]
  %   A: A function handle such that A(x) is the sensing matrix times x
  %   At: A function handle such that A(x) is the transposed sensing matrix
  %       times x, divided by the square of the norm  of the matrix.
  % Output:
  %   img: The normalized image
  %   raw_img: The output of the reconstruction algorithm, before, normalization
  %           
  
  if ~isfield(para, 's_model')
    para.s_model = 1;
  end
  if ~isfield(para, 'L_threshold')
    para.L_threshold = 20;
  end
  
  warning off;
  par = Set_parameters(para.CSr, para.L_threshold, para.s_model);
  par.s_model = para.s_model;
  par.h = para.row;
  par.w = para.col;
  par.y = gather(y)*255;  % prevent GPU processing
  
  raw_img = NLR_CS_Reconstruction_had( par, A, At );
  raw_img = uint8(round(max(min(raw_img,255),0)));
  img = im2double(raw_img);
  %img = raw_img/max(raw_img(:));
end

