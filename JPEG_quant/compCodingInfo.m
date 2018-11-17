function coding_info = compCodingInfo(enc_opts)
  %compCodingInfo computes a struct which cotains all the parameters necessary to
  %    compute encode, decode and reconstruct the measurements.
  %
  % INPUT:
  %   enc_opts: A CS_EncImg_Params object
  % OUTPUT:
  %   coding_info: A struct with the following fields:
  %      enc_opts: The input enc_opts
  %      sens_mtrx: The sensing matrix object
  %      q_prms: A struct of parameters used for creating the quantizer
  %      quantizer: The quantizer (object of class UniformQuantizer)
  
  % Determine single or double precision
   CompMode.setDefaultUseSingle(enc_opts.use_single);
   
  % Generate sensing matrix
  mt_args = enc_opts.msrmnt_mtrx.args;
  mt_args.n_cols = enc_opts.blk_size(1) * enc_opts.blk_size(2);
  mt_args.dims = enc_opts.blk_size(1:2);
  mt_args.rnd_seed = enc_opts.random.seed; 
  if isfield(enc_opts.msrmnt_mtrx, 'nxt')
    nxt = enc_opts.msrmnt_mtrx.nxt;
  else
    nxt = [];
  end
  
  sens_mtrx = genMsrmntMatrix(enc_opts.msrmnt_mtrx.type, mt_args, ...
    enc_opts.msrmnt_input_ratio, nxt);
  
  if enc_opts.process_color
    sens_mtrx = SensingMatrixKron.construct({SensingMatrixUnit(3), sens_mtrx});
  end

  % Definitions for quantizer
  n_no_clip = sens_mtrx.nNoClip();
  msrs_noise = sens_mtrx.calcMsrsNoise(sens_mtrx.nCols(), 2^(-enc_opts.bit_depth)); 
  q_prms = struct(...
    'msrs_noise', sens_mtrx.toCPUFloat(msrs_noise), ... (digitization noise)
    'mean', [], ... Compute mean from measurements
    'stdv', [], ... Compute std. Dev. from measurements
    'n_no_clip', n_no_clip, ...
    'n_clip', sens_mtrx.nRows() - n_no_clip ...
    );
  quantizer = UniformQuantizer(enc_opts, q_prms);

  % definitions for lossless coding
  if ~enc_opts.process_color
    vtp = 'BW_8';
  else
    vtp = 'YUV444_8';
  end
  raw_vid = RawVidInfo(struct(...
    'fps', 1,...
    'height', enc_opts.blk_size(1), ...
    'width', enc_opts.blk_size(2),  ...
    'type', vtp ...
    ));
  vid_blocker = VidBlocker(struct('yb_size', enc_opts.blk_size), raw_vid);
  vid_region = VidRegion([1,1,1],  vid_blocker);
   
  coding_info = struct(...
    'enc_opts', enc_opts, ...
    'sens_mtrx', sens_mtrx, ...
    'q_prms', q_prms, ...
    'quantizer', quantizer, ...
    'vid_region', vid_region);

end

