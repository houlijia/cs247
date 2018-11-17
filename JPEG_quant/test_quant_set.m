function [stats, enc_opts, dec_opts, raw_img, raw_nq_img] = ...
    test_quant_set( image_files, enc_opts, dec_opts, unquant, ...
    verbose, out_dir, parproc )
  %test_quant_set runs test_quant with different options on multiple files
  %
  %INPUT: 
  %  image_files: A cell array of input file names or a single input file.
  %  enc_opts: An appropriate first argument for ProcessingParams.getCases().
  %    Therefore, it can be one of the following:
  %    - A character string beginning with '<'. The characters past the '<'
  %      specify a file name containing a JSON string which defines a sequence
  %      of enc_opts structs.
  %    - A character string not beginning with '<'. The string contains a JSON 
  %      string which defines a sequence of enc_opts structs.
  %    - An array of structs or an array of cell arrays containing structs
  %      each specifing conditions
  %  dec_opts: An appropriate first argument for ProcessingParams.getCases().
  %    Therefore, it can be one of the following:
  %    - A character string beginning with '<'. The characters past the '<'
  %      specify a file name containing a JSON string which defines a sequence
  %      of enc_opts structs.
  %    - A character string not beginning with '<'. The string contains a JSON 
  %      string which defines a sequence of enc_opts structs.
  %    - An array of structs or an array of cell arrays containing structs
  %      each specifing conditions
  %    If dec_opts is empty it is replaced by struct('alg','GAP_TV').
  %  unquant: If true, decode also using unquantized measurements.
  %  verbose: Print information about progress of processing
  %  out_dir: If not specified or empty, no output files are written. Otherwise,
  %    it is the directory for output files. An asterisk (*) in out_dir is
  %    replaced by a date string (only the first one).
  %  parproc: If true, use parallel processing on images. Default=false
  %  
  %OUTPUT:
  %  stats: an array of structs containing statistics for each
  %         image/enc_opts/dec_opts ombination.
  %   enc_opts: a cell array containing all the expanded options of enc_opts
  %   dec_opts: a cell array containing all the expanded options of dec_opts
  %   raw_img: a cell array containing the raw reconstructed images. Same size
  %   a     stats.
  %   raw_nq_img: a cell array containing raw reconstructed images from
  %               unquantized measurements. Same size as stats.
  
  mex_clnp = initMexContext();   %#ok (prevent error on unused variable)
  
  if ~iscell(image_files)
    image_files = {image_files};
  end
  
  if unquant
    hdr_fmt = [...
      '+-------------------+---------------+------+------+------+-----+-----+-----+-----+-----+-----+--------+------+------+-------+\n'...
      '|    Image name     |     Opts      | file | Msrs | Hist | N   |SSIM |PSNR |SSIM |PSNR | CS  |Nrm.Qntz|bits/ |MsrSNR|%% Sat. |\n'...
      '|                   |     index     | size | size | size | Msrs|     |     |NQ   |NQ   |ratio|Interval|UsdMsr|  dB  | msrs  |\n'...
      '+-------------------+---------------+------+------+------+-----+-----+-----+-----+-----+-----+--------+------+------+-------+\n'...
      ];
    stt_fmt = ...
      '|%19s|%15s|%6d|%6d|%6d|%5d|%5.3f|%5.1f|%5.3f|%5.1f|%5.3f|%9.4g|%6.1f|%6.3f|%7.3f|\n';
    sep_fmt = ...
      '+-------------------+---------------+------+------+------+-----+-----+-----+-----+-----+-----+--------+------+------+-------+\n';
  else
    hdr_fmt = [...
      '+-------------------+---------------+------+------+------+-----+-----+-----+-----+--------+------+------+-------+\n'...
      '|    Image name     |     Opts      | file | Msrs | Hist | N   |SSIM |PSNR | CS  |Nrm.Qntz|bits/ |MsrSNR|%% Sat. |\n'...
      '|                   |     index     | size | size | size | Msrs|     |     |ratio|Interval|UsdMsr|  dB  | msrs  |\n'...
      '+-------------------+---------------+------+------+------+-----+-----+-----+-----+--------+------+------+-------+\n'...
      ];
    stt_fmt = ...
      '|%19s|%15s|%6d|%6d|%6d|%5d|%5.3f|%5.1f|%5.3f|%9.4g|%6.1f|%6.3f|%7.3f|\n';
    sep_fmt = ...
      '+-------------------+---------------+------+------+------+-----+-----+-----+-----+--------+------+------+-------+\n';
  end

  if isempty(dec_opts)
    dec_opts = struct('alg', 'GAP_TV');
  end
  
  if nargin < 7
    parproc = false;
  end
  
  [enc_opts,enc_opt_str] = ProcessingParams.getCases(enc_opts);
  [dec_opts,dec_opt_str] = ProcessingParams.getCases(dec_opts);
  
  n_enc_opts = numel(enc_opts(:));
  n_dec_opts = numel(dec_opts(:));
  n_img = numel(image_files(:));
    
  stats = struct(...
    'img_sz', cell(n_enc_opts, n_dec_opts, n_img), ... no. of color pixels in image
    'n_msrs', cell(n_enc_opts, n_dec_opts, n_img), ... no. of measurements in image
    'n_sat_msrs', cell(n_enc_opts, n_dec_opts, n_img), ... no. of saturated measurements
    'n_byte', cell(n_enc_opts, n_dec_opts, n_img), ... no. of byte in coded image
    'n_byte_msrs', cell(n_enc_opts, n_dec_opts, n_img), ... no. of byte in coded measurements
    'n_byte_hist', cell(n_enc_opts, n_dec_opts, n_img), ... no. of byte in coded histogram
    'msrs_db_rms', cell(n_enc_opts, n_dec_opts, n_img), ... ratio of RMS quantization error to
                                    ... RMS of measurements, in dB.
    'img_psnr', cell(n_enc_opts, n_dec_opts, n_img), ... PSNR of image
    'img_ssim', cell(n_enc_opts, n_dec_opts, n_img) ... SSIM of image
    );
  
  if nargout > 3
    raw_img = cell(size(stats));
    if nargout > 4
      raw_nq_img = cell(size(stats));
    end
  end
  
  if verbose >= 1
    fprintf('Encoding Options:\n');
    for iopt = 1:n_enc_opts
      fprintf('%s:\t%s\n', enc_opt_str{iopt},...
        show_str(enc_opts(iopt), struct(),...
        struct('prefix', '    ','cell_marked', true, 'struct_marked', ...
        true, 'struct_sep', '; ')));
    end
    fprintf('Decoding Options:\n');
    for iopt = 1:n_dec_opts
      fprintf('%s:\t%s\n', dec_opt_str{iopt},...
        show_str(dec_opts(iopt), struct(),...
        struct('prefix', '    ','cell_marked', true, 'struct_marked', ...
        true, 'struct_sep', '; ')));
    end
  end
  
  if nargin < 6
    out_dir = '';
  elseif ~isempty(out_dir)
    out_dir = regexprep(out_dir, '*',...
      datestr(now,'yyyymmdd_HHMM'),'once');
    if ~exist(out_dir, 'dir')
      mkdir(out_dir);
    end
    fprintf('Output directory: %s\n', out_dir);
  end
  
  if verbose >=1
    fprintf(hdr_fmt);
  end
  switch nargout
    case 5
      if parproc
        parfor iimg = 1:n_img
          [stats(:, :, iimg), raw_img(:,:,iimg), raw_nq_img(:,:,iimg)] =...
            test_quant_set_loop(image_files{iimg}, enc_opts, ...
            dec_opts, out_dir,...
            enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose);
        end
      else
        for iimg = 1:n_img
          [stats(:, :, iimg), raw_img(:,:,iimg), raw_nq_img(:,:,iimg)] =...
            test_quant_set_loop(image_files{iimg}, enc_opts, ...
            dec_opts, out_dir,...
            enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose);
        end
      end
    case 4
      if parproc
        parfor iimg = 1:n_img
          [stats(:, :, iimg), raw_img(:,:,iimg)] =...
            test_quant_set_loop(image_files{iimg}, enc_opts, ...
            dec_opts, out_dir,...
            enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose);
        end
      else
        for iimg = 1:n_img
          [stats(:, :, iimg), raw_img(:,:,iimg)] =...
            test_quant_set_loop(image_files{iimg}, enc_opts, ...
            dec_opts, out_dir,...
            enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose);
        end
      end
    otherwise
      if parproc
        parfor iimg = 1:n_img
          stats(:, :, iimg) =...
            test_quant_set_loop(image_files{iimg}, enc_opts, ...
            dec_opts, out_dir,...
            enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose);
        end
      else
        for iimg = 1:n_img
          stats(:, :, iimg) =...
            test_quant_set_loop(image_files{iimg}, enc_opts, ...
            dec_opts, out_dir,...
            enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose);
        end
      end
  end
  
  if verbose >= 1
    fprintf(sep_fmt);
  end
  
  if ~isempty(out_dir)
    results_out = fullfile(out_dir, 'results.mat');
    save(results_out, 'image_files','enc_opts', 'dec_opts', 'stats');
    fprintf('%s. Done. Output directory: %s\n', datestr(datetime), out_dir);
  else
    fprintf('%s. Done.\n', datestr(datetime));
  end
  
 end

