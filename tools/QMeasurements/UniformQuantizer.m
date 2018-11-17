classdef UniformQuantizer < CompMode
  %UniformQuantizer Provides functions to quantize and unquantize sets of
  %measurements
  %   Detailed explanation goes here
  
  properties

    % The optimal dequantizer does not quantize to the middle of the bin
    % but a little closer to the 0 because the guassian distribution is not
    % even throughout the bin.  If this is true, measurement will be
    % corrected accordingly.  A preliminary test showed that this caused a
    % slight degradation, so this option is disabled.
    correct_gaus_unquant = false;

    % Quantization step q_wdth is determined by q_wdth_mltplr.  If
    % q_wdth_mltplr==0 there is no quantization (q_wdth=0, meaning that values
    % are left as is, getting their integer value). If q_wdth_mltplr > 0
    % the qunatization interval is
    %    q_wdth = q_wdth_mltplr * q_wdth_unit
    % where q_wdth_unit is the quantization step for which the
    % variance of the quantization  error is the same as the variance of
    % the digitization error of the pixels as it appears in the
    % measurement (q_wdth_unit==0 means not set yet).
    % Each pixel was derived by digitizing the analog video
    % samples into integer pixel values, hence the variance of the
    % digitization error is 1/12.  In the measurements, the average
    % digitization error is (1/m)*sum(a(i,j)^2, i=1:m, j=1:n)*(1/12) where a(i,j)
    % are the entries of the sensing matrix, which is of dimension m by n.
    % The variance of the quantization is q_wdth^2/12.  Therefore,
    %   normalized_q_wdth = sqrt((1/m)*sum(a(i,j)^2, i=1:m, j=1:n))
    % In particular, if the entries of the sensing matrix have a magnitude
    % of 1, then normalized_q_wdth = sqrt(n).
    
    q_wdth_mltplr=0;
    q_wdth_mode = CS_EncParams.Q_WDTH_NRML;
    q_wdth_rng_exp = [];
    
    % The quantizer clips all values which exceeds a certain magnitude. This
    % magnitude is given by q_ampl_mltplr times the standard deviation of the
    % measurements.
    q_ampl_mltplr=0;
    
    % A flag indicating whether clipped values should be saved and returned as
    % a part of the quantization operation.
    save_clipped = false;
    
    % If non-zero, standard deviation is computed by sorting the measurements,
    % removing the fraction at the ends that should be discarded based on
    % Guassian distribution assumptions, computing the variance based on the
    % remaining measurement and correcting for Gaussian distribution only.
    only_in_rng_mltplr = 0;
    
    % Factor to multiply by the standard deviation computed if
    % only_in_rng_mltplr is non-zero
    only_in_rng_factor = 1;

    
    % pointer to cast function to the right accuracy and processor
    toLabel;
    
  end
    
  properties(SetAccess = private)
    % Parameters for quantization given as an array of structs.
    % If n_parblk==0, the array is of size [blk_cnt(1), blk_cnt(2), 1]. If
    % n_parblk>0, the array is of size [blk_cnt(1), blk_cnt(2), n_parfrm].
    % The fields in the struct are:
    %
    %  msrs_noise - measurements noise (Quantizer's CPUFloat)
    %  intvl - quantization interval (as computed by UniformQuantizer.cmpIntvl)
    %  mean - a vaule to use as the mean of the measurements, [] = compute the 
    %         mean from the measurements.
    %  stdv - a value to use as the standard deviation of the measurements
    %         instead of a calculated value. [] = compute the standard deviation
    %         from the measurements. -1 = Compute the standard deviation from
    %         the measurements which are within the sppecified range: Suppose 
    %         the measurements are x(1),...,x(N), sorted in an increasing order.
    %         Let d=q_ampl_mltplr. Compute D such that D/N=(1-F(d))/2, where F()
    %         is a Gaussian distribution with zero mean and variance of one. Then
    %         compute an estimate of the Gaussian variance from x(1+D),...,x(N-D)
    %         by:
    %            variance = var(x(1+D:N-D))/C((F(A)-F(-A))-2*A*f(A))
    %         where
    %            C = 2*(F(d) - 0.5 - d*f(d)) = erf(A)-2*A*exp(-A^2)/sqrt(pi)
    %         where f() is the PDF corresponding to F and
    %            A=d/sqrt(2)
    %         In this cae, if mean is not specified or [], the mean is computed as
    %         the mean of x(1+D:N-D)
    %  n_clip - no. of clipped measurements in the block.
    %  n_no_clip - no. of no clip measurements.
    %  clip_offset - if n_parblk > 0 this is the offset from beginning of
    %         clipped measurements to beginning of clipped measurements for the
    %         current block. Otherwise it is 0.
    %  no_clip_offset - if n_parblk > 0 this is the offset from beginning of
    %                   measurements to beginning of no-clip measurements for this
    %                   block. Otherwise it is 0.
    %    Note: n_clip, n_no_clip, clip_offset and no_clip_offset are of type
    %    UniformQuantizer.toCPUIndex()
    %
    q_params;
  end
  
  properties(Access = private)
    q_wdth_sq12=0; % q_wdth_mltplr * sqrt(12)
    mex_func;

    %max number of measurements in a block
    n_msrs_max = 0;
  end
  
  methods
    % Constructor
    % INPUT:
    %   enc_opts - an object of class CS_EncParams, or a struct that has the
    %     following fields:
    %       qntzr_wdth_mltplr - value to be assigned to q_wdth_mltplr
    %       qntzr_ampl_stddev - value to be assigned to q_ampl_mltplr
    %       obj.qntzr_outrange_action - if equals CS_EncParams.Q_SAVE the quantizer will save 
    %                clipped values.
    %       qntzr_only_in_rng - if true mean and standard deviation are computed after
    %                 discardig measurements not in range.
    %       par_blks - if non zero process par_blks frames at a time.
    %       use_single - if true use single precision. Otherwise, double
    %                    precision.
    %   prms - an array of structs, each corresponding to one block, with the 
    %       following fields:
    %       - msrs_noise - measurements noise (standard deviation)
    %       - mean - a vaule to use as the mean of the measurements, [] =
    %                compute the mean from the measurements. 
    %       - stdv - a value to use as the standard deviation of the
    %                measurements, instead of a calculated value. 
    %                [] = compute the standard deviation from the measurements. 
    %                -1 = Compute the standard deviation from  the measurements 
    %                     which are within the sppecified range (see comment for
    %                     q_params above).
    %       - n_clip - no. of clipped measurements in the block.
    %       - n_no_clip - no. of no clip measurements.
    %   use_gpu - if true, use GPU. If missing default is used (use GPU if
    %             available)
    %       
    function obj = UniformQuantizer(enc_opts, prms, use_gpu)
      if nargin < 3
        use_gpu = CompMode.defaultUseGpu();
      end
      obj@CompMode(use_gpu, enc_opts.use_single);
      obj.q_ampl_mltplr = enc_opts.qntzr_ampl_stddev;
      obj.q_wdth_mode = enc_opts.qntzr_wdth_mode;
      obj.q_wdth_mltplr = enc_opts.qntzr_wdth_mltplr;
      switch obj.q_wdth_mode
        case CS_EncParams.Q_WDTH_CSR
          obj.q_wdth_mltplr = obj.q_wdth_mltplr / enc_opts.msrmnt_input_ratio;
        case CS_EncParams.Q_WDTH_RNG
          obj.q_wdth_rng_exp = enc_opts.qntzr_wdth_rng_exp;
        case CS_EncParams.Q_WDTH_RNG_CSR
          obj.q_wdth_rng_exp = enc_opts.qntzr_wdth_rng_exp;
          obj.q_wdth_mltplr = obj.q_wdth_mltplr / enc_opts.msrmnt_input_ratio;
      end          
      obj.save_clipped = (enc_opts.qntzr_outrange_action == CS_EncParams.Q_SAVE);
      obj.setCastFloat();  % must be before obj.setQntzrOnlyInRng()
      obj.setQntzrOnlyInRng(enc_opts.qntzr_only_in_rng);
      
      % set q_params
      [nvb, nhb, t_max] = size(prms);
      obj.q_params = struct(...
        'n_clip', cell(nvb, nhb, t_max),...
        'n_no_clip', cell(nvb, nhb, t_max),...
        'msrs_noise', cell(nvb, nhb, t_max),...
        'intvl', cell(nvb, nhb, t_max),...
        'mean', cell(nvb, nhb, t_max),...
        'stdv', cell(nvb, nhb, t_max),...
        'clip_offset', cell(nvb,nhb,t_max),...
        'no_clip_offset', cell(nvb,nhb,t_max));
      
      clip_offset = 0;
      no_clip_offset = 0;
      for t=1:t_max
        for h=1:nhb
          for v=1:nvb
            obj.q_params(v,h,t).n_clip = obj.toCPUIndex(prms(v,h,t).n_clip);
            obj.q_params(v,h,t).n_no_clip = obj.toCPUIndex(prms(v,h,t).n_no_clip);
            n_msrs = obj.q_params(v,h,t).n_clip + obj.q_params(v,h,t).n_no_clip;
            if n_msrs > obj.n_msrs_max
              obj.n_msrs_max = n_msrs;
            end
            obj.q_params(v,h,t).mean = prms(v,h,t).mean;
            obj.q_params(v,h,t).stdv = prms(v,h,t).stdv;
            obj.q_params(v,h,t).msrs_noise = obj.toCPUFloat(prms(v,h,t).msrs_noise);
            obj.q_params(v,h,t).intvl = ...
              obj.toCPUFloat(obj.cmpIntvl(obj.q_params(v,h,t).msrs_noise));
            obj.q_params(v,h,t).clip_offset = clip_offset;
            obj.q_params(v,h,t).no_clip_offset = no_clip_offset;
            if enc_opts.par_blks > 0
              clip_offset = clip_offset + obj.q_params(v,h,t).n_clip;
              no_clip_offset = no_clip_offset + obj.q_params(v,h,t).n_no_clip;
            end
          end
        end
      end 
    end
      
    function intvl = cmpIntvl(obj, msrs_noise)
      switch obj.q_wdth_mode
        case CS_EncParams.Q_WDTH_ABS
          intvl = obj.q_wdth_mltplr;
        case {CS_EncParams.Q_WDTH_NRML, CS_EncParams.Q_WDTH_CSR}
          intvl = msrs_noise * obj.q_wdth_sq12;
        case {CS_EncParams.Q_WDTH_RNG, CS_EncParams.Q_WDTH_RNG_CSR}
          % In this cse the interval computed here is a minimum bound. The
          % actual value is comptued based on the measurements.
          intvl = msrs_noise * sqrt(12);
        otherwise
          error('Unexpected value of obj.q_wdth_mode: %d', obj.q_wdth_mode);
      end
      intvl = obj.toCPUFloat(intvl);
    end
        
    function intvl = cmpRngIntvl(obj, prms, msrs)
      if obj.q_wdth_mode == CS_EncParams.Q_WDTH_RNG || ...
          obj.q_wdth_mode == CS_EncParams.Q_WDTH_RNG_CSR
        n_msrs = obj.toCPUFloat(length(msrs));
        if n_msrs > 0
          intvl = (sum(msrs .^ obj.q_wdth_rng_exp)/n_msrs) ^ (1/obj.q_wdth_rng_exp);
          intvl = obj.toCPUFloat(intvl);
          intvl = intvl * obj.q_wdth_mltplr;
          intvl = max(intvl, prms.intvl);
        else
          intvl = prms.intvl;
        end
      else
        intvl = prms.intvl;
      end
    end
    
    function qmsr = quantizeBlk(obj, msrs_no_clip, msrs_clip, blk_indx)
      qmsr = obj.quantize(msrs_no_clip, msrs_clip, obj.q_params(blk_indx(1),blk_indx(2),1));
    end
    
    function qmsr = quantizeFrms(obj, msrs_no_clip, msrs_clip, n_frms)
      qmsr = obj.quantize(msrs_no_clip, msrs_clip, obj.q_params(:,:,1:n_frms));
    end
    
    function  qmsr = quantize(obj, msrs_no_clip, msrs_clip, params)
      % quantize - quantize an array of measurements into a single
      % QuantMeasurements object or a cell array of QuantMeasurements objects.
      % Input:
      %   obj - the quantizer object
      %   msrs_no_clip - the data array to of no-clip measurements.
      %   msrs_clip - the data array of clip measurements
      %   params  - An array of structs of parameters. The fields are:
      %     The following fields are arrays of n_blk elements if n_blk is
      %     specified or scalars
      %       msrs_noise - measurements noise (Quantizer's CPUFloat)
      %       intvl - quantization interval (as computed by UniformQuantizer.cmpIntvl)
      %               if q_mode is Q_WDTH_RNG or Q_WDTH_RNG_CSR this is only the
      %               lower bound for the actual interval.
      %       mean - a vaule to use as the mean of the measurements, [] = unspecifed.
      %       stdv - a value to use as the standard deviation of the measurements
      %         instead of a calculated value. [] = unspecifed. A value of -1 has a
      %         special meaning:  Suppose the measurements are x(1),...,x(N) and let
      %         d=q_ampl_mltplr. Compute D such that D/N=(1-F(d))/2, where F() is a
      %         Gaussian distribution with zero mean and variance of one. Then
      %         compute the estimate of the Gaussian variance from x(1+D),...,x(N-D)
      %         by:
      %            variance = var(x(1+D:N-D))/C((F(A)-F(-A))-2*A*f(A))
      %         where
      %            C = 2*(F(d) - 0.5 - d*f(d)) = erf(A)-2*A*exp(-A^2)/sqrt(pi)
      %         where f() is the PDF corresponding to F and
      %            A=d/sqrt(2)
      %         In this cae, if mean is not specified or [], the mean is computed as
      %         the mean of x(1+D:N-D)
      %       n_clip - no. of clipped measurements in the block.
      %       n_no_clip - no. of no clip measurements.
      %       clip_offset - (only if n_blk is defined) offset from beginning of
      %         clipped measurements to beginning of clipped measurements for
      %         this block.
      %       no_clip_offset - (only if n_blk is defined) offset from beginning of
      %         measurements to beginning of no-clip measurements for this block
      % Output
      %   qmsr - an object of type QuantMeasurements or a cell array of QM
      %          Objects.
      
      % Determine mean and standard deviation
      qmsr = QuantMeasurements();
      msrs_no_clip = obj.toFloat(msrs_no_clip);
      msrs_clip = obj.toFloat(msrs_clip);
      params = params(:);
      qmsr.params = params;
      if ~obj.only_in_rng_mltplr
        if isempty(params(1).stdv)
          if isempty(params(1).mean)
            [qmsr.mean_msr, qmsr.stdv_msr] = arrayfun(@(p)...
              obj.compMeanStdv(msrs_clip(p.clip_offset+1:p.clip_offset+p.n_clip)),...
              params);
          else
            qmsr.stdv_msr = arrayfun(@(p) ...
              obj.compStdv(msrs_clip(p.clip_offset+1:p.clip_offset+p.n_clip), p.mean), ...
              params);
            qmsr.mean_msr = [params.mean]';
          end
        else
          if isempty(params(1).mean)
            qmsr.mean_msr = arrayfun(@(p)...
              obj.compMean(msrs_clip(p.clip_offset+1:p.clip_offset+p.n_clip)), ...
              params);
          else
            qmsr.mean_msr = [params.mean]';
          end
          qmsr.stdv_msr = [params.stdv]';
        end
      else
        if isempty(params(1).stdv)
          if isempty(params(1).mean)
            [qmsr.mean_msr, qmsr.stdv_msr] = arrayfun(@(p) obj.compMeanStdvInRng(...
              msrs_clip(p.clip_offset+1:p.clip_offset+p.n_clip)) ,params);
          else
            qmsr.stdv_msr = arrayfun(@(p) obj.compStdvInRng(...
              msrs_clip(p.clip_offset+1:p.clip_offset+p.n_clip), p.mean), params) ;
            qmsr.mean_msr = [params.mean]';            
          end
        else
          if isempty(params(1).mean)
            qmsr.mean_msr = arrayfun(@(p) obj.compMeanInRng(...
              msrs_clip(p.clip_offset+1:p.clip_offset+p.n_clip)), params);
          else
            qmsr.mean_msr = [params.mean]';
          end
          qmsr.stdv_msr = [params.stdv]';
        end
      end
      
      [qmsr.n_bins, bnc, bc, sv, q_intvl] = ...
        arrayfun(@comp_quant, params, qmsr.mean_msr, qmsr.stdv_msr);
      qmsr.bin_no_clip = vertcat(bnc{:});
      qmsr.bin_numbers = vertcat(bc{:});
      qmsr.saved = obj.toSInt(vertcat(sv{:}));
      qmsr.params = params;
      if obj.q_wdth_mode == CS_EncParams.Q_WDTH_RNG || ...
          obj.q_wdth_mode == CS_EncParams.Q_WDTH_RNG_CSR
        qmsr.intvl = q_intvl;
      end

      function [n_bins, bin_nc, bin_c, sav, q_intvl] = comp_quant(prms, mn, sd)
        q_intvl = obj.cmpRngIntvl(prms,...
          msrs_clip(prms.clip_offset+1:prms.clip_offset+prms.n_clip));

        % Compute interval and amplitude
        max_bin = obj.toCPUIndex(...
          round((sd * obj.q_ampl_mltplr) ./q_intvl));
        n_bins = obj.toCPUIndex(2*max_bin + 1);
        ampl = (obj.toCPUFloat(max_bin) + 0.5) .* q_intvl;
        offset = ampl - mn;
        
        if obj.save_clipped
          [bin_nc, bin_c, sav] = obj.mex_func.quant(...
            msrs_no_clip(prms.no_clip_offset+1:prms.no_clip_offset+prms.n_no_clip),...
            msrs_clip(prms.clip_offset+1:prms.clip_offset+prms.n_clip),...
            q_intvl, offset, n_bins);
        else
          [bin_nc, bin_c] = obj.mex_func.quant(...
            msrs_no_clip(prms.no_clip_offset+1:prms.no_clip_offset+prms.n_no_clip),...
            msrs_clip(prms.clip_offset+1:prms.clip_offset+prms.n_clip),...
            q_intvl, offset, n_bins);
          sav = [];
        end
        bin_nc = {bin_nc};
        bin_c = {bin_c};
        sav = {sav};
      end
      
%       % Code for comparing the mex functions against Matlab implementation
%       ref_bin_no_clip = ceil((offset + msrmnts(1:qmsr.n_no_clip))/qmsr.intvl);
%       if obj.use_gpu
%         if obj.save_clipped
%           [ref_bin_numbers, qm] = arrayfun(@quantMsrSave, ...
%             msrmnts(n_no_clip+1:end), ...
%             obj.toFloat(offset), ...
%             obj.toFloat(qmsr.intvl), ...
%             obj.toFloat(n_bins));
%           ref_saved = qm(ref_bin_numbers == (n_bins+1));
%         else
%           ref_bin_numbers = arrayfun(@quantMsr, ...
%             msrmnts(n_no_clip+1:end), ...
%             obj.toFloat(offset), ...
%             obj.toFloat(qmsr.intvl), ...
%             obj.toFloat(n_bins));
%           ref_saved = [];
%         end
%       else
%         mtx = obj.ones(size(msrmnts(n_no_clip+1:end)));
% 
%         if obj.save_clipped
%           [ref_bin_numbers, qm] = arrayfun(@quantMsrSave, ...
%             msrmnts(n_no_clip+1:end), ...
%             obj.toFloat(offset) * mtx, ...
%             obj.toFloat(qmsr.intvl) * mtx, ...
%             obj.toFloat(n_bins) * mtx);
%           ref_saved = qm(qmsr.bin_numbers == (n_bins+1));
%         else
%           ref_bin_numbers = arrayfun(@quantMsr, ...
%             msrmnts(n_no_clip+1:end), ...
%             obj.toFloat(offset) * mtx, ...
%             obj.toFloat(qmsr.intvl) * mtx, ...
%             obj.toFloat(n_bins) * mtx);
%           ref_saved = [];
%         end
%       end
%       
%       % there may be an occasional mismatch due to numeric error
%       % differences
%       if ~isequal(ref_bin_no_clip, qmsr.bin_no_clip) ||...
%           ~isequal(ref_bin_numbers, qmsr.bin_numbers) ||...
%           ~isequal(ref_saved, qmsr.saved)
%         if norm(double(ref_bin_no_clip) - double(qmsr.bin_no_clip), inf) <= 1 ||...
%             norm(double(ref_bin_numbers) - double(qmsr.bin_numbers),inf) <= 1
%           warning('small quantization mismatch');
%         else
%           error('quantization mismatch');
%         end
%       end
    end
     
    function [msrmnts, clipped_indices, intvl] = unquantize(obj, qmsr)
      % unquantize - reconstruct values from quantization bin numbers
      % Input:
      %   obj - the quantizer object
      %   qmsr - an object of type QuantMeasurements
      %   q_unit - (optional) overrides obj.q_wdth_unit
      %   blk_indx - indices of blocks, size [n_blk,3]
      % output
      %   msrmnts - Reconstructed measurements
      %   clipped_indices - Indices of measurement which could not be
      %                  reconstructed because they were clipped.
      %   intvl - the quantization interval
      %
      mean_msr = obj.toCPUFloat(qmsr.mean_msr(:));
      stdv_msr = obj.toCPUFloat(qmsr.stdv_msr(:));
      n_bins = obj.toCPUIndex(qmsr.n_bins(:));
      saved = obj.toFloat(qmsr.saved(:));
      
      function [bin_numbers, clip_ind, clip_ind_ofst] = ...
          get_blk_bin_numbers(ofst,len, nc_ofst, nc_len, nbn)
        bin_numbers =  { obj.toFloat(qmsr.bin_numbers(ofst+1:ofst+len)) };
        clip_ind = find(bin_numbers{1} > nbn);
        clip_ind = { obj.toCPUIndex(clip_ind) };   %#ok
        clip_ind_ofst = { clip_ind{1} + ...
          obj.toCPUIndex(ofst) + obj.toCPUIndex(nc_ofst) + obj.toCPUIndex(nc_len) };
      end
      
      [bin_numbers, clip_ind, clip_ind_ofst] = arrayfun(@get_blk_bin_numbers,...
        [qmsr.params(:).clip_offset]', [qmsr.params(:).n_clip]', ...
        [qmsr.params(:).no_clip_offset]', [qmsr.params(:).n_no_clip]', n_bins);
      
      bin_no_clip = arrayfun(...
        @(ofst,len) {obj.toFloat(qmsr.bin_no_clip(ofst+1:ofst+len))},...
        [qmsr.params.no_clip_offset]', [qmsr.params.n_no_clip]');
      
      % Fill clipped measurements with saved measurements, if available
      if ~isempty(saved)
        i_save = 0;
        for iblk=1:length(n_bins)
          c_ind = clip_ind{iblk};
          if isempty(c_ind)
            continue;
          end
          not_fixed = true(size(c_ind));
          for k=1:length(c_ind)
            if bin_numbers{iblk}(c_ind(k)) == n_bins(iblk)+1;
              i_save = i_save + 1;
              if saved(i_save) > 0
                bin_numbers{iblk}(c_ind(k)) = saved(i_save) + ...
                  obj.toFloat(n_bins(iblk));
              else
                bin_numbers{iblk}(c_ind(k)) = saved(i_save);
              end
              not_fixed(k) = false;
            end
          end
          clip_ind_ofst{iblk} = clip_ind_ofst{iblk}(not_fixed);
          clip_ind{iblk} = clip_ind{iblk}(not_fixed);
        end
      end
      
      max_bin = obj.toCPUIndex((n_bins-1)/2);
      switch obj.q_wdth_mode
        case {CS_EncParams.Q_WDTH_RNG, CS_EncParams.Q_WDTH_RNG_CSR}
          intvl = qmsr.intvl;
        otherwise
          intvl = [qmsr.params.intvl]';
      end
      ampl = (obj.toCPUFloat(max_bin) + 0.5) .* intvl;
      offset = ampl - mean_msr;

      function [msrs_no_clip, msrs_clip] = unquant_blk(bnc, bc, clip_ind, ivl, ...
          ofst, sd)
        msrs_no_clip = { bnc{1} * ivl - ofst };
        msrs_clip = bc{1} * ivl - ofst;
        if obj.correct_gaus_unquant && stdv_msr > 0
          msrs_clip = ...
          obj.correctGausUnquant(msrs_clip, bc, n_bins, ivl, sd);
        end
        msrs_clip(clip_ind{1}) = 0;
        msrs_clip = { msrs_clip };
      end
      
      [b_no_clip, b_clip] = arrayfun(@unquant_blk, bin_no_clip, bin_numbers, ...
        clip_ind, intvl, offset, stdv_msr);
      
      % Compute measurements and clipped indices.
      msrmnts = [vertcat(b_no_clip{:}); vertcat(b_clip{:})];
      
      % clipped_indices are indices of clipped measurements after unsorting.
      clipped_indices = vertcat(clip_ind_ofst{:});
    end
  end
  
   methods (Access=protected)
    
    function setQntzrOnlyInRng(obj, ok)
      % Set parameters for estimating mean and variance. If ok is false,
      % estimation is done using all measurements. If it is false, estimation is
      % done only with the measuremetns within the range and corrected for a
      % full Gaussian distribution.
      if ok
         E = erf(double(obj.q_ampl_mltplr)/sqrt(2));
         obj.only_in_rng_mltplr = obj.toCPUFloat(0.5*(1-E));
         obj.only_in_rng_factor = obj.toCPUFloat(1/sqrt(E));
      else
        obj.only_in_rng_mltplr = obj.toCPUFloat(0);
        obj.only_in_rng_factor = obj.toCPUFloat(1);
      end
    end
    
    function mn = compMean(obj, msrs)
      mn =  obj.mex_func.mean(msrs);
        
%       % Test code - compare MEX with Matlab results
%       ref_mean = obj.toCPUFloat(mean(msrs));
%       if abs(mn - ref_mean)/(1E-10 + ref_mean) > 1E-10
%         error('Error in mean: Matlab - Mex= %g -%g = %g',...
%           ref_mean, mn, ref_mean - mn)
%       end
    end
    
    function mn = compMeanInRng(obj, msrs)
      D = obj.toCPUIndex(floor(double(length(msrs)) *...
        double(obj.only_in_rng_mltplr)));
      msrs = sort(msrs);
      mn = obj.mex_func.mean(msrs(D+1:end-D));
    end
    
    function sd = compStdv(obj, msrs, mn)
      n_vec = length(msrs);
      if  n_vec > obj.toCPUFloat(1)
        nvc1 = n_vec - obj.toCPUFloat(1);
        sqrdf = obj.mex_func.sub_sqr(mn, msrs);
        sd = obj.mex_func.mean(sqrdf) * (n_vec/nvc1);
        sd = obj.toCPUFloat(sqrt(double(sd)));
        
%         % Test code - compare MEX with Matlab results
%         mn_sqr_ref = (obj.toCPUFloat(sum(msrs .^ 2)) - n_vec*mn*mn)/nvc1;
%         ref_stdv = obj.toCPUFloat(sqrt(mn_sqr_ref));
%         if abs(sd - ref_stdv)/(1E-10 + ref_stdv) > 1E-10
%           error('Error in standard deviation: Matlab - Mex= %g -%g = %g',...
%             ref_stdv, sd, ref_stdv - sd)
%         end
      else
        sd = obj.toCPUFloat(0);
      end
    end
    
    function sd = compStdvInRng(obj, msrs, mn)
      D = obj.toCPUIndex(floor(double(length(msrs)) *...
        double(obj.only_in_rng_mltplr)));
      n_vec = obj.toCPUFloat(length(msrs) - (D+D));
      if  n_vec > obj.toCPUFloat(1)
        msrs = sort(msrs);
        nvc1 = n_vec - obj.toCPUFloat(1);
        sqrdf = obj.mex_func.sub_sqr(mn, msrs(D+1:end-D));
        sd = obj.mex_func.mean(sqrdf) * (obj.only_in_rng_factor *n_vec/nvc1);
        sd = obj.toCPUFloat(sqrt(double(sd)));
        
%         % Test code - compare MEX with Matlab results
%         mn_sqr_ref = (obj.toCPUFloat(sum(msrs .^ 2)) - n_vec*mn*mn)/nvc1;
%         ref_stdv = obj.toCPUFloat(sqrt(mn_sqr_ref));
%         if abs(sd - ref_stdv)/(1E-10 + ref_stdv) > 1E-10
%           error('Error in standard deviation: Matlab - Mex= %g -%g = %g',...
%             ref_stdv, sd, ref_stdv - sd)
%         end
      else
        sd = obj.toCPUFloat(0);
      end
    end
    
    function [mn,sd] = compMeanStdv(obj, msrs)
      [mn,sd] = obj.mex_func.mean_stdv(msrs);
%       % Test code - compare MEX with Matlab results
%       ref_mean = obj.compMean(msrs);
%       if abs(mn - ref_mean)/(1E-10 + ref_mean) > 1E-10
%         error('Error in mean: Matlab - Mex= %g -%g = %g',...
%           ref_mean, mn, ref_mean - mn)
%       end
%       
%       ref_stdv = obj.compStdv(msrs, ref_mean);
%       if abs(sd - ref_stdv)/(1E-10 + ref_stdv) > 1E-10
%         error('Error in standard deviation: Matlab - Mex= %g -%g = %g',...
%           ref_stdv, sd, ref_stdv - sd)
%       end
   end
   
    function [mn,sd] = compMeanStdvInRng(obj, msrs)
      D = obj.toCPUIndex(floor(double(length(msrs)) *...
        double(obj.only_in_rng_mltplr)));
      msrs = sort(msrs);
      [mn,sd] = obj.mex_func.mean_stdv(msrs(D+1:end-D));
      sd = sd * obj.only_in_rng_factor;
    end
    
    function setCastFloat(obj)
      obj.setCastFloat@CompMode();
      
      if obj.use_gpu
        obj.mex_func = struct(...
          'sub_sqr', @cuda_sub_sqr_mex,...
          'mean', @cuda_mean_mex,...
          'mean_stdv', @cuda_mean_stdv_mex,...
          'quant', @cuda_quant_mex...
          );
        
        obj.toLabel = @UniformQuantizer.toGPULabel;
      else
        obj.mex_func = struct(...
          'sub_sqr', @cc_sub_sqr_mex,...
          'mean', @cc_mean_mex, ...
          'mean_stdv', @cc_mean_stdv_mex,...
          'quant', @cc_quant_mex...
        );
        
        obj.toLabel = @UniformQuantizer.toCPULabel;
      end
      
      obj.q_ampl_mltplr = obj.toCPUFloat(obj.q_ampl_mltplr);
      obj.q_wdth_mltplr = obj.toCPUFloat(obj.q_wdth_mltplr);
      obj.q_wdth_rng_exp = obj.toCPUFloat(obj.q_wdth_rng_exp);
      obj.q_wdth_sq12 = obj.toCPUFloat(double(obj.q_wdth_mltplr)*sqrt(12));
      obj.only_in_rng_mltplr = obj.toCPUFloat(obj.only_in_rng_mltplr);
      obj.only_in_rng_factor = obj.toCPUFloat(obj.only_in_rng_factor);
    end
  end
  
  methods (Static)
    function uq = makeUniformQuantizer(enc_opts, blkr, mtrx_mgr, use_gpu)
      % This function generates and returns a uniform quantizer.
      % INPUT:
      %   enc_opts - an object of class CS_EncParams, or a struct that has the
      %     following fields:
      %       qntzr_wdth_mltplr - value to be assigned to q_wdth_mltplr
      %       qntzr_ampl_stddev - value to be assigned to q_ampl_mltplr
      %       qntzr_outrange_action - if equals CS_EncParams.Q_SAVE the quantizer will save
      %                clipped values.
      %       qntzr_only_in_rng - if true mean and standard deviation are computed after
      %                 discardig measurements not in range.
      %       par_blks - if non zero process par_blks frames at a time.
      %   blkr - a VidBlocker object
      %   mtrx_mgr - an object of type SensingMatrixManager
      %   use_gpu - if true, use GPU
      %
      
      % set prms
      nvb = blkr.blk_cnt(1);
      nhb = blkr.blk_cnt(2);
      t_max = max(enc_opts.par_blks, 1);
      prms = struct(...
        'n_clip', cell(nvb,nhb,t_max),...
        'n_no_clip', cell(nvb,nhb,t_max),...
        'msrs_noise', cell(nvb,nhb,t_max),...
        'mean', cell(nvb,nhb,t_max),...
        'stdv', cell(nvb,nhb,t_max));
      
      for t=1:t_max
        for h=1:nhb
          for v=1:nvb
            blk_indx = [v,h,t];
            vid_region = VidRegion(blk_indx, blkr, ...
              [enc_opts.zero_ext_b; enc_opts.zero_ext_f; enc_opts.wrap_ext]);
            sens_mtrx = mtrx_mgr.getBlockMtrx(blk_indx, vid_region);
            n_msrs = sens_mtrx.nRows();
            prms(v,h,t).n_no_clip = sens_mtrx.nNoClip();
            prms(v,h,t).n_clip = n_msrs - prms(v,h,t).n_no_clip;
            prms(v,h,t).msrs_noise = sens_mtrx.calcMsrsNoise(vid_region.n_orig_blk_pxls);
          end
        end
      end
      uq = UniformQuantizer(enc_opts, prms, use_gpu);
    end
    
    function x = toCPULabel(x)
      x = int16(full(x));
    end
    
    function x = toGPULabel(x)
      x = gpuArray(int16(full(x)));
    end
    
    % Compute the source entropy, assuming that it is Guassian with sigma=1
    % (otherwise step has to be divided by sigma).
    %   Input
    %     max_bin - maximal bin number. Bin numbers range from -max_bin+1
    %         to max_bin, where max_bin-1 is the highest unsaturated bin
    %         and max_bin indicates saturation.  max_bin can be a row
    %         vector, in which case results are calculate for each value in
    %         the array.
    %     step - quantization step size. can be a column vector, in which
    %     case results are calculated for each column in the vector
    %     stddev - Standard deviation of measurements (optional, default=1)
    %   Output.  All outuputs are arrays of the same size as max_bin, with
    %   corresponding values.
    %
    %     entrp - The entropy of the quantization labels.
    %     use_prob - probability that the label is not saturated
    %     use_entrp - entropy per used measurment:  entrp/use_prob.
    function [entrp, use_prob, use_entrp] = compGausEntropy(max_bin,step, stddev)
      if nargin > 2
        step = step /stddev;
      end
      n_step = length(step);
      n_max = max(max_bin);
      n_vals = length(max_bin);
      
      % Allocate arrays;
      entrp = builtin('zeros',n_step,n_vals); 
      if nargout > 1
        use_prob = builtin('zeros',n_step,n_vals);
        if nargout > 2
          use_entrp = builtin('zeros',n_step,n_vals);
        end
      end
      
      % Compute values
      for istp = 1:n_step
        pts = (step(istp)/sqrt(2))*(0.5:1:(n_max-0.5));
        vals = erf(pts);
        dff = diff([0 vals]);
        dff2 =  dff;
        dff2(2:end) = dff2(2:end)*0.5;
        ents = -dff .* log(dff2);
        ents(isnan(ents))=0;
        
        for k=1:n_vals
          k_max = max_bin(k);
          used_p = vals(k_max);
          unused_p = 1- used_p;
          ent_sat = - unused_p * log(unused_p);
          if isnan(ent_sat); ent_sat = 0; end
          entrp(istp,k) = sum(ents(1:k_max)) + ent_sat;
          if nargout > 1
            use_prob(istp,k) = used_p;
            if nargout > 2
              use_entrp(istp,k) = entrp(istp,k)/used_p;
            end
          end
        end
      end
    end
    
    % Compute the source entropy, assuming that it is Guassian with sigma=1
    % (otherwise step has to be divided by sigma).
    %   Input
    %     max_bin - maximal bin number. Bin numbers range from -max_bin+1
    %         to max_bin, where max_bin-1 is the highest unsaturated bin
    %         and max_bin indicates saturation.  max_bin can be a row
    %         vector, in which case results are calculate for each value in
    %         the vector.
    %     ampl - quantization amplitude. It can be a column vector, in which
    %         case results are calculate for each value in
    %         the vector.
    %   Output.  All outuputs are arrays of the same size as max_bin, with
    %   corresponding values.
    %
    %     entrp - The entropy of the quantization labels.
    %     use_prob - probability that the label is not saturated
    %     use_entrp - entropy per used measurment:  entrp/use_prob.
    function [entrp, use_prob, use_entrp] = compGausEntropyAmpl(max_bin,ampl)
      entrp = builtin('zeros',length(ampl), length(max_bin));
      use_prob = entrp;
      use_entrp = entrp;
      for j=1:length(ampl)
        for k=1:length(max_bin)
          [e,u,ue] =...
            UniformQuantizer.compGausEntropy(max_bin(k), ...
            ampl(j)/(0.5+max_bin(k)));
          entrp(j,k) = e;
          use_prob(j,k) = u;
          use_entrp(j,k) = ue;
        end
      end
    end
    
    
    % Compute the entropy, per used bin, assuming that the source is
    % Guassian with sigma=1(otherwise step has to be divided by sigma).
    %   Input
    %     max_bin - maximal bin number. Bin numbers range from -max_bin+1
    %         to max_bin, where max_bin-1 is the highest unsaturated bin
    %         and max_bin indicates saturation.  max_bin can be a row
    %         array, in which case results are calculate for each value in
    %         the array.
    %     step - quantization step size.
    %   Output.  All outuputs are arrays of the same size as max_bin, with
    %   corresponding values.
    %     use_entrp - entropy per used measurment/ array of the same size
    %                 as max_bin, with corresponding values.
    function use_entrp = compGausEntropyPerUsed(max_bin,step)
      [~,~,use_entrp] = UniformQuantizer.compGausEntropy(max_bin,step);
    end
    
    % Correct the unquantized measurements by taking into account that the
    % distribution in each bin is Gaussian, not uniform
    % Input:
    %   msrments - measurements computed based on Gaussian assumption
    %   qmsr - The unquantized measurements, assuming uniform behavior
    %   intvl - step size
    %   n_bins - number of unsaturated bins
    % Output
    %   msrmnts - corrected measurements
    function msrmnts = correctGausUnquant(msrmnts, bin_numbers, ...
        n_bins, intvl, stdv_msr)
      max_bin = (n_bins-1)/2;
      step = intvl/stdv_msr;
      step2 = 0.5*step*step;
      denom = 0.5 * diff(erf((step/sqrt(2))*(0.5:max_bin+0.5)));
      inds = 1:max_bin;
      nom = (sqrt(2/pi))*exp(-step2*(inds.*inds+0.25)).*sinh(step2*inds);
      crct = ((nom./denom) - inds*step)*stdv_msr;
      crct = [-crct(max_bin:-1:1), 0, crct, ...
        builtin('zeros', 1, (n_bins+1:max(bin_numbers)), 'like', crct)];
      msrmnts(bin_inds) = msrmnts(bin_inds) + crct(bin_numbers(bin_inds))';
    end
  end
  
  methods (Static, Access=protected)
   end
end

