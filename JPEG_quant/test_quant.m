function [q_code, qm, rec_q, orig_img, y, qy, sat_ind, rec_q_raw, rec_o, rec_o_raw] = ...
    test_quant(image_file, enc_opts, dec_opts, verbose, ...
    cs_file, rec_q_file, rec_o_file)
  % test_quant runs a quantization test
  % 
  % INPUT
  %   image_file: Input image file name
  %   enc_opts - (optional): either a CS_EncImgParams object or a struct
  %     containing options which measurements generation encoding and decoding.
  %     Any field which is not specified gets a default value, specified in the
  %     struct def_opts in getImageCodingInfo(). 
  %   dec_opts: (optional) Should be a struct or a cell array of structs, where
  %             each struct specifies a reconstruction algorithm. Each struct
  %             should have at least the field 'alg', which species the
  %             algorithm. Other algorithm parameters may be specified as well.
  %             If not present or empty, set to struct('alg', 'GAP_TV').
  %   verbose - (optional): verbosity level (default=1).
  %     0 - print nothing
  %     1 - print quantization stats
  %     2 - print supplied enc_opts and quantizations stats and display images.
  %     3 - print final enc_opts and quantization stats and display images.
  %   cs_file: If present and not empty, output file for compressive measurements.
  %   rec_q_file: (optional) If present and not empty, output file for 
  %               image(s) reconstructed from quantized measurements.  
  %               If dec_opts is a struct or not
  %               present, this should be a string with the file specifications.
  %               If dec_opts is a cell array, req_q_file should be a cell array
  %               of the same size, containing file names for each reconstructed
  %               image.
  %   rec_o_file: (optional) If present and not empty, output file for 
  %               image(s) reconstructed from original (unquantized) measurements.  
  %               If dec_opts is a struct or not
  %               present, this should be a string with the file specifications.
  %               If dec_opts is a cell array, req_q_file should be a cell array
  %               of the same size, containing file names for each reconstructed
  %               image.
  % OUTPUT
  %   q_code: A byte array (uint8) containing the coded image (content of
  %       cs_file).
  %   qm: QuantMeasurements objects containing the measurements and related
  %       information.
  %   rec_q: If dec_opts is a struct or not present, this is an array containing 
  %          the reconstructed image. Otherwise it is a cell array, with each
  %          cell containing the image reconstructed according to the
  %          corresponding dec_opts.
  %   orig_img: Original image as a floating point array
  %   y: Unquantized measurements
  %   qy: Quantized measurements
  %   sat_ind: Indices of saturated measurements
  %  rec_q_raw: The raw decoded image before normalization. Same structure as
  %             rec_q
  %   rec_o: If dec_opts is a struct or not present, this is an array containing 
  %          the image reconstructed from the original (unquantized) 
  %          measurements. Otherwise it is a cell array, with each
  %          cell containing the image reconstructed according to the
  %          corresponding dec_opts.
  %  rec_o_raw: The raw decoded image before normalization. Same structure as
  %             rec_o
 
  % Initialization for MEX
  mex_clnp = initMexContext();   %#ok (prevent error on unused variable)
  
  % Parse input arguments
  if nargin < 2 || isempty(dec_opts)
    dec_opts = struct('alg', 'GAP_TV');
  end
  if nargin < 3
    enc_opts = struct();
  end
  if nargin < 4
    verbose = 1;
  end
  if nargin < 5
    cs_file = '';
  end
  if nargin < 6
    rec_q_file = '';
  end
  if nargin < 7
    rec_o_file = '';
  end
  
  do_noquant = (nargout >= 7 && ~isempty(rec_o_file));
  % parse enc_opts
  
  if verbose == 2
    fprintf('----------------\nSupplied enc_opts:\n%s\n----------------\n', ...
      show_str(enc_opts));
  end
  
  % Encode
  [orig_img, y, qme, q_code, enc_info] = test_encode(image_file, enc_opts, cs_file);
  
  qm = qme;

  if verbose >= 2
    figure()
    for k=1:numel(dec_opts)
      subplot(numel(dec_opts),2+do_noquant,1+(k-1)*numel(dec_opts));
      imshow(orig_img);
      if k==1
        title(sprintf('Original %d x %d', size(orig_img,1), size(orig_img,2)));
      end
    end
  end
      
  if verbose == 3
    fprintf('----------------\nFinal enc_opts = \n%s\n----------------\n', ...
      show_str(enc_info.enc_opts));
  end
  
  % Reconstruct from original measurements
  if do_noquant
    [rec_o, rec_o_raw] = CSImgReconstruct(y, enc_info, dec_opts);
    if ~isempty(rec_o_file)
      if ~iscell(dec_opts)
        rec_o = {rec_o};
        rec_o_raw = {rec_o_raw};
        rec_o_file = {rec_o_file};
      end
      for k=1:numel(dec_opts)
        o_img = rec_o{k} / max([rec_o{k}(:);1E-10]);
        if isa(o_img, 'gpuArray')
          o_img = gather(o_img);
        end
        imwrite(o_img, rec_o_file{k})
        rec_o_raw_file = regexprep(rec_o_file{k},'[.][^.]*$', '-raw_rec.mat');
        raw_rec_img = rec_o_raw{k};   %#ok
        save(rec_o_raw_file, 'raw_rec_img');
      end
      if ~iscell(dec_opts)
        rec_o = rec_o{1};
      end
    end
    if verbose
      if ~iscell(dec_opts)
        d_opts = {dec_opts};
        rec_o = {rec_o};
      else
        d_opts = dec_opts;
      end
      for k=1:numel(d_opts)
        fprintf('%s without quantization: ssim=%f psnr=%f\n', d_opts{k}.alg,...
        ssim(rec_o{k},orig_img),psnr(rec_o{k},orig_img));
      end
      fprintf('%s\n', qme.report(enc_info));
      
      if verbose >= 2
        for k=1:numel(d_opts)
          subplot(1,3,3+(k-1)*numel(dec_opts));
          imshow(rec_o{k});
          title(sprintf('%s CSr=%.2f', d_opts{k}.alg, ...
            enc_info.enc_opts.msrmnt_input_ratio));
        end
      end
      if ~iscell(dec_opts)
        rec_o = rec_o{1};
      end
    end
  end
  
  if nargin >= 2 && ~isempty(cs_file)
    src = cs_file; 
  else
    src  = q_code;
  end

  % Decode
  [rec_q, qy, sat_ind, qmd, dec_info, rec_q_raw] = test_decode(src,  dec_opts, rec_q_file);
%   % debugging version
%   [rec_q, qy, sat_ind, qmd, dec_info] = test_decode(src, dec_opts, rec_q_file, ...
%     struct('enc_opts', enc_info.enc_opts, 'qm', qme));
  
  % Verify decoding correctness
  if ~qme.isEqual(qmd)
    error('qmd and qme are not identical');
  end
  
  % modify sens matrix to allow correct comparison
  d_info = dec_info;
  d_info.sens_mtrx.setZeroedRows([]);
  msr_mtrx = d_info.enc_opts.msrmnt_mtrx;
  if ~isfield(msr_mtrx,'args') || isempty(msr_mtrx.args)
    msr_mtrx.args = struct();
  end
  if ~isfield(msr_mtrx,'nxt')
    msr_mtrx.nxt = [];
  end
  d_info.enc_opts.setParams(struct('msrmnt_mtrx', msr_mtrx));
  d_info.sens_mtrx = d_info.sens_mtrx.copy();
  d_info = rmfield(d_info, 'q_max_err');
  d_info = rmfield(d_info, 'q_stddev_err');
  if ~isEqual(d_info, enc_info)
    error('dec_info and enc_info do not match identical');
  end
  
  if verbose
    % Report quantization error
    y_ref = y;
    y_ref(sat_ind) = 0;
    n_ref = length(y_ref)-length(sat_ind);
    
    err = sqrt(sum((qy-y_ref) .^ 2 )/n_ref);
    ampl = sqrt(sum(y_ref .^ 2 )/n_ref);
    qnr = 20*log10(err / dec_info.q_prms.msrs_noise);
    fprintf('  Ampl:Quant error (RMS) = %12.4E : %12.4E (%6.1f dB), QNR=%6.1f dB\n',...
      err, ampl, 20*log10(ampl/(err + 1E-10)), qnr);

    if ~iscell(dec_opts)
      d_opts = {dec_opts};
      rec_q = {rec_q};
    else
      d_opts = dec_opts;
    end
    for k=1:numel(d_opts)
      fprintf('%s with quantization: ssim=%f psnr=%f\n', d_opts{k}.alg, ...
        ssim(rec_q{k},orig_img), psnr(rec_q{k},orig_img));

      if verbose >= 2
        subplot(1,2+do_noquant,2+(k-1)*numel(dec_opts));
        imshow(rec_q{k});
        title(sprintf('%s qstep=%.1f', d_opts{k}.alg,...
          dec_info.enc_opts.qntzr_wdth_mltplr));
      end
    end
    if ~iscell(dec_opts)
      rec_q = rec_q{1};
    end
  end
  
end
