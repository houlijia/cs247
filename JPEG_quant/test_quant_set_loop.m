function [stats, raw_img, raw_nq_img] = test_quant_set_loop(img_file, enc_opts, dec_opts, ...
    out_dir, enc_opt_str, dec_opt_str, hdr_fmt, stt_fmt, sep_fmt, unquant, verbose)
  % RunCS encoding and decoding tests on a file
  %  
  % INPUT:
  %   img_file: path to the image file
  %   enc_opts: A cell or struct array of file specifications
  %   dec_opts: A struct or a cell array of structs cotaining decoder
  %     specification. Those specfications should at least include the fild
  %     'alg' indicating the type of reconstruction algorithm. If dec_opts is
  %     empty it is replaced by struct('alg','GAP_TV').
  %   out_dir: If not empty, CS files and reconstructed files are written into
  %            here
  %   enc_opt_str: A cell array of same size as enc_opts, containing identifying 
  %            strings for each encoding specification.
  %   dec_opt_str: A cell array of same size as dec_opts, containing identifying 
  %            strings for each decoding specification.
  %   hdr_fmt: A string containing the header of the statistics text output
  %   stt_fmt: Format for statistics printing
  %   sep_fmt: Format of separator line at the end
  %   unquant: If true, decode also using unquantized measurements.
  %   verbose: Verbosity level
  % OUTPUT:
  %   stats - a struct array of statiscs results
  %   raw_img - a cell array containing the raw reconstructed images
  %   raw_nq_img - a cell array containing raw reconstructed images from
  %               unquantized measurements.
  
  if isempty(dec_opts)
    dec_opts = struct('alg', 'GAP_TV');
  end
  
  d_opts = dec_opts;
  if ~iscell(dec_opts)
    dec_opts = {dec_opts};
  end
  n_enc_opts = numel(enc_opts(:));
  n_dec_opts = numel(dec_opts(:));
  
  [~,img_name,~] = fileparts(img_file);

  if ~isempty(out_dir)
    img_out_dir = fullfile(out_dir, img_name);
    [status, err_msg, err_id] = mkdir(img_out_dir);
    if ~status
      error(err_id, 'Failed to create folder %s: %s', img_out_dir, err_msg);
    end

    txt_out = fopen(fullfile(img_out_dir, 'results.txt'),'wt');
    csv_out = fopen(fullfile(img_out_dir, 'results.csv'),'wt');
    results_out = fullfile(img_out_dir, 'results.mat');
    if nargout > 1
      raw_img_file = fullfile(img_out_dir, 'raw.mat');
      if nargout > 2
        raw_nq_img_file = fullfile(img_out_dir, 'raw-nq.mat');
      else
        raw_nq_img_file = [];
      end
    else
      raw_img_file = [];
      raw_nq_img_file = [];
    end
    image_files = {img_file}; %#ok

  else
    txt_out = [];
    csv_out = [];
    results_out = [];
  end
  
  if ~isempty(txt_out)
    fprintf(txt_out, 'Encoding Options:\n');
    for iopt = 1:n_enc_opts
      if  iscell(enc_opts)
        e_opt = enc_opts{iopt};
      else
        e_opt = enc_opts(iopt);
      end
      fprintf(txt_out, '%s:\n%s\n', enc_opt_str{iopt},...
        show_str(e_opt, struct(),...
        struct('prefix', '    ','cell_marked', true, 'struct_marked', true)));
    end
    fprintf(txt_out, 'Decoding Options:\n');
    for iopt = 1:n_dec_opts
      fprintf(txt_out, '%s:\n%s\n', dec_opt_str{iopt},...
        show_str(dec_opts{iopt}, struct(),...
        struct('prefix', '    ','cell_marked', true, 'struct_marked', true)));
    end
    fprintf(txt_out, hdr_fmt);
  end
  if ~isempty(csv_out)
    fprintf(csv_out, ...
      ['Image name, Opts index, file size, SSIM, PSNR, CS ratio, bits/UsdMsr, '...
      'MsrSnr dB, %% Lost msrs\n']);
  end

  stats = struct(...
    'img_sz', cell(n_enc_opts, n_dec_opts), ... no. of color pixels in image
    'n_msrs', cell(n_enc_opts, n_dec_opts), ... no. of measurements in image
    'n_sat_msrs', cell(n_enc_opts, n_dec_opts), ... no. of saturated measurements
    'n_byte', cell(n_enc_opts, n_dec_opts), ... no. of byte in coded image
    'n_byte_msrs', cell(n_enc_opts, n_dec_opts),... no. of bytes in coded measurements
    'n_byte_hist', cell(n_enc_opts, n_dec_opts),... no. of bytes in coded histogram
    'msrs_db_rms', cell(n_enc_opts, n_dec_opts), ... ratio of RMS quantization error to
    ... RMS of measurements, in dB.
    'img_psnr', cell(n_enc_opts, n_dec_opts), ... PSNR of image
    'img_ssim', cell(n_enc_opts, n_dec_opts) ... SSIM of image
    );
  
  if nargout > 1
    raw_img = cell(size(stats));
    if nargout > 2
      raw_nq_img = cell(size(stats));
    end
  end
  
  for i_opt = 1:n_enc_opts
    if iscell(enc_opts)
      enc_opt = enc_opts{i_opt};
    else
      enc_opt = enc_opts(i_opt);
    end
    
    if ~isempty(out_dir)
      out_name = [img_name '-' enc_opt_str{i_opt}];
      cs_file = fullfile(img_out_dir, [out_name '.cs']);
      rec_q_file = cell(1, n_dec_opts);
      if unquant
        rec_o_file = cell(1, n_dec_opts);
      end
      for k=1:n_dec_opts
        rec_q_file{k} = fullfile(img_out_dir,...
          [img_name '-' get_opt_str(i_opt,k) '.png']);
        if unquant
          rec_o_file{k} = fullfile(img_out_dir,...
            [img_name get_opt_str(i_opt,k) '-nqnt.png']);
        end
      end
    end
    
    if unquant
      switch nargout
        case 3
          [q_code, qm, rec_q, img_data, y, qy, sat_ind, rec_q_raw,...
            rec_o, rec_o_raw] =...
            test_quant(img_file, enc_opt, dec_opts, 0, cs_file, rec_q_file, rec_o_file);
          
          for k=1:n_dec_opts
            if isa(rec_q{k}, 'gpuArray')
              rec_q{k} = gather(rec_q{k});
            end
            if isa(rec_q_raw{k}, 'gpuArray')
              rec_q_raw{k} = gather(rec_q_raw{k});
            end
            if isa(rec_o{k}, 'gpuArray')
              rec_o{k} = gather(rec_o{k});
            end
            if isa(rec_o_raw{k}, 'gpuArray')
              rec_o_raw{k} = gather(rec_o_raw{k});
            end
          end
          raw_nq_img(i_opt,:) = rec_o_raw(:)';
          raw_img(i_opt,:) = rec_q_raw(:)';
          
        case 2
          [q_code, qm, rec_q, img_data, y, qy, sat_ind, rec_q_raw,...
            rec_o] =...
            test_quant(img_file, enc_opt, dec_opts, 0, cs_file, rec_q_file, rec_o_file);
          
          for k=1:n_dec_opts
            if isa(rec_q{k}, 'gpuArray')
              rec_q{k} = gather(rec_q{k});
            end
            if isa(rec_q_raw{k}, 'gpuArray')
              rec_q_raw{k} = gather(rec_q_raw{k});
            end
            if isa(rec_o{k}, 'gpuArray')
              rec_o{k} = gather(rec_o{k});
            end
          end
          raw_img(i_opt,:) = rec_q_raw(:)';
          
        otherwise
          [q_code, qm, rec_q, img_data, y, qy, sat_ind, rec_o] =...
            test_quant(img_file, enc_opt, dec_opts, 0, cs_file, rec_q_file, rec_o_file);
          
          for k=1:n_dec_opts
            if isa(rec_q{k}, 'gpuArray')
              rec_q{k} = gather(rec_q{k});
            end
            if isa(rec_o{k}, 'gpuArray')
              rec_o{k} = gather(rec_o{k});
            end
          end
      end
    else
      switch nargout
        case {2,3}
          [q_code, qm, rec_q, img_data, y, qy, sat_ind, rec_q_raw] =...
            test_quant(img_file, enc_opt, dec_opts, 0, cs_file, rec_q_file);
          
          for k=1:n_dec_opts
            if isa(rec_q{k}, 'gpuArray')
              rec_q{k} = gather(rec_q{k});
            end
            if isa(rec_q_raw{k}, 'gpuArray')
              rec_q_raw{k} = gather(rec_q_raw{k});
            end
          end
          raw_img(i_opt,:) = rec_q_raw(:)';
          
        otherwise
          [q_code, qm, rec_q, img_data, y, qy, sat_ind] =...
            test_quant(img_file, enc_opt, dec_opts, 0, cs_file, rec_q_file);
          
          for k=1:n_dec_opts
            if isa(rec_q{k}, 'gpuArray')
              rec_q{k} = gather(rec_q{k});
            end
          end
      end
    end
    
    if isa(img_data, 'gpuArray')
      img_data = gather(img_data);
    end
    
    % Compute mearements quantization error, in dB
    y(sat_ind) = 0;
    qy(sat_ind) = 0;
    err = norm(qy-y)+1E-10;
    ampl = norm(y)+1E-10;
    dbrms = 20*log10(ampl/err);
    
    stt = struct(...
      'img_sz', numel(img_data), ...
      'n_msrs', qm.nMsrs(), ...
      'n_sat_msrs', numel(sat_ind) + numel(qm.saved), ...
      'n_byte', length(q_code), ...
      'n_byte_msrs', qm.len_enc_msrs,...
      'n_byte_hist', qm.len_enc_hist,...
      'msrs_db_rms', dbrms, ...
      'img_psnr', 0, ...
      'img_ssim', 0);

    for k=1:n_dec_opts
      stt.img_psnr = psnr(rec_q{k}, img_data);
      stt.img_ssim = ssim(rec_q{k}, img_data);
      stats(i_opt,k) = stt;
      if unquant
        if isa(rec_o{k}, 'gpuArray')
          rec_o{k} = gather(rec_o{k});
        end
        img_nq_psnr = psnr(rec_o{k}, img_data);
        img_nq_ssim = ssim(rec_o{k}, img_data);
      end
      
      if verbose >= 1
        if unquant
          fprintf(stt_fmt,...
            img_name,  get_opt_str(i_opt,k), stt.n_byte, stt.n_byte_msrs, stt.n_byte_hist,...
            stt.n_msrs, stt.img_ssim, stt.img_psnr,...
            img_nq_ssim, img_nq_psnr, ...
            double(stt.n_msrs)/double(stt.img_sz), ...
            double(qm.getIntvl()) /(double(qm.stdv_msr) + 1E-20), ...
            8*stt.n_byte/(double(stt.n_msrs)-double(stt.n_sat_msrs)), ...
            stt.msrs_db_rms, ...
            stt.n_sat_msrs*100/stt.n_msrs);
        else
          fprintf(stt_fmt,...
            img_name,  get_opt_str(i_opt,k), stt.n_byte, stt.n_byte_msrs, stt.n_byte_hist,...
            stt.n_msrs, stt.img_ssim, stt.img_psnr,...
            double(stt.n_msrs)/double(stt.img_sz), ...
            double(qm.getIntvl()) / (double(qm.stdv_msr) + 1E-20), ...
            8*stt.n_byte/(double(stt.n_msrs)-double(stt.n_sat_msrs)), ...
            stt.msrs_db_rms, ...
            stt.n_sat_msrs*100/stt.n_msrs);
        end
      end
      
      if ~isempty(txt_out)
        if unquant
          fprintf(txt_out, stt_fmt,...
            img_name,  get_opt_str(i_opt,k), stt.n_byte, stt.n_byte_msrs, stt.n_byte_hist,...
            stt.n_msrs, stt.img_ssim, stt.img_psnr,...
            double(stt.n_msrs)/double(stt.img_sz), ...
            double(qm.getIntvl()), ...
            8*stt.n_byte/(double(stt.n_msrs)-double(stt.n_sat_msrs)), ...
            stt.msrs_db_rms, ...
            stt.n_sat_msrs*100/stt.n_msrs);
        else
          fprintf(txt_out, stt_fmt,...
            img_name,  get_opt_str(i_opt,k), stt.n_byte, stt.n_byte_msrs, stt.n_byte_hist,...
            stt.n_msrs, stt.img_ssim, stt.img_psnr,...
            double(stt.n_msrs)/double(stt.img_sz), ...
            double(qm.getIntvl()), ...
            8*stt.n_byte/(double(stt.n_msrs)-double(stt.n_sat_msrs)), ...
            stt.msrs_db_rms, ...
            stt.n_sat_msrs*100/stt.n_msrs);
        end
      end
      
      if ~isempty(csv_out)
        fprintf(csv_out, '%15s,%15s,%6d,%5.3f,%5.1f,%5.3f,%6.3f,%6.1f,%6.3f\n',...
          img_name,  get_opt_str(i_opt,k), stt.n_byte, stt.img_ssim, stt.img_psnr,...
          double(stt.n_msrs)/double(stt.img_sz), ...
          8*stt.n_byte/(double(stt.n_msrs)-double(stt.n_sat_msrs)), ...
          stt.msrs_db_rms, ...
          stt.n_sat_msrs*100/stt.n_msrs);
      end
      if ~isempty(results_out)
        save(results_out, 'image_files', 'enc_opts', 'dec_opts', 'stats');
      end
    end
  end
  
  if ~isempty(results_out)
    save(results_out, 'image_files', 'enc_opts', 'dec_opts', 'stats');
  end
  if ~isempty(raw_img_file)
    save(raw_img_file, 'raw_img');
  end
  if ~isempty(raw_nq_img_file)
    save(raw_nq_img_file, 'raw_nq_img');
  end
  
  if ~isempty(txt_out)
    fprintf(txt_out, '%s\n', sep_fmt);
    fclose(txt_out);
  end
  if ~isempty(csv_out)
    fclose(csv_out);
  end
  
  function opt_str = get_opt_str(i_enc,i_dec)
    if ~iscell(d_opts) || n_dec_opts == 1 || isempty(dec_opt_str{i_dec})
      opt_str = enc_opt_str{i_enc};
    else
      opt_str = [enc_opt_str{i_enc} '~' dec_opt_str{i_dec}];
    end
  end
    
  
end
