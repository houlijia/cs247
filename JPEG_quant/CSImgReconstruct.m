function [img, raw_img] = CSImgReconstruct(y, coding_info, para)
  % CSImgReconstruct reconstructs an image from the measurements
  % INPUT:
  %   y: measurements vector
  %   coding_info: the coding parameters
  %   para: Optional struct or cell array of containing structs
  %         Each struct should have the field
  %           alg: reconstruction algorithm name.
  %         If para is not specified it is set to struct('alg','GAP_TV')
  % OUTPUT:
  %   img: Reconstructed image. If para is a struct (or not defined) img is
  %        the image. If para is a cell array then  img is a cell array of
  %        images.
  %   raw_img: The output of the reconstruction algorithm, before,
  %        normalization. If para is a struct (or not defined) raw_img is
  %        the image. If para is a cell array then  img is a cell array of
  %        images.
  
  
  enc_opts = coding_info.enc_opts;
  if enc_opts.process_color
    error('CSImgReconstruct:BadArgs', ...
      'No support for reconstruction of color images');
  end
  
  if nargin < 3
    para = struct('alg','GAP_TV');
  end
  
  if isstruct(para)
    para_cel = {para};
  else
    para_cel = para;
  end
  
  A = @(x) coding_info.sens_mtrx.multVec(x);
  At = @(x) coding_info.sens_mtrx.multTrnspVec(x) / coding_info.sens_mtrx.norm()^2;
  
  img = cell(size(para_cel));
  raw_img = cell(size(para_cel));
  
  for k=1:numel(para_cel)
    para_cel{k}.row = enc_opts.blk_size(1);
    para_cel{k}.col = enc_opts.blk_size(2);
    para_cel{k}.CSr = enc_opts.msrmnt_input_ratio;
    if isfield(coding_info, 'q_max_err')
      para_cel{k}.q_max_err = coding_info.q_max_err;
    else
      para_cel{k}.q_max_err = 0;
    end
    if isfield(coding_info, 'q_stddev_err')
      para_cel{k}.q_stddev_err = coding_info.q_stddev_err;
    else
      para_cel{k}.q_stddev_err = 0;
    end
    
    
    switch para_cel{k}.alg
      case 'GAP_TV'
        [img{k}, r_img] = CSImgReconstruct_GAP_TV(y, para_cel{k}, A, At );
      case 'GAP_BM3D'
        [img{k}, r_img] = CSImgReconstruct_GAP_BM3D(y, para_cel{k}, A, At );
      case 'DAMP'
          if(strcmp(coding_info.enc_opts.msrmnt_mtrx.type, 'MD_DCT')||strcmp(coding_info.enc_opts.msrmnt_mtrx.type, 'DCT'))
             [img{k}, r_img] = CSImgReconstruct_DAMP_DCT(y, para_cel{k}, A, At );
          else
              [img{k}, r_img] = CSImgReconstruct_DAMP(y, para_cel{k}, A, At );
          end
      case 'NLR_CS'
        [img{k}, r_img] = CSImgReconstruct_NLR_CS(y, para_cel{k}, A, At );
      case 'SLOPE'
        [img{k}, r_img] = CSImgReconstruct_SLOPE(y, para_cel{k}, A, At );
      otherwise
        error('Unknown algoithm: %s', para_cel{k}.alg);
    end
    
    if nargout > 0
      img{k} = single(img{k});
      if isstruct(para) && isscalar(para)
        img = img{1};
      end
    end
    
    if nargout > 1
      raw_img{k} = single(r_img);
      if isstruct(para) && isscalar(para)
        raw_img = raw_img{1};
      end
    end
  end
  
  if isstruct(para)
      if iscell(img)
        img = img{1};
        raw_img = raw_img{1};
      end
  end

end