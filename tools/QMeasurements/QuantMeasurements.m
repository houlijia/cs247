classdef QuantMeasurements < CodeElement
  %QuantMeasurements contains the output of quantization of measurements
  
  properties(Constant)
    PROB_MLTPLR = 1e6;
    min_stdv = 1e-10;   % Minimum standard deviation
    
    force_use_gaus = false;
  end
  
  properties
    % The first n_no_clip measurements are divided by the quantization 
    % interval, but are not clipped. These first n_no_clip measurements
    % also do not participate in mean and standard deviation computation

    bin_no_clip=[];   % The no clip bins.
    bin_numbers=[];   % an array of bin numbers. corresponding to the
                      % measurements.
    saved=[];         % array of quantized values corresponding to clipped
                      % values. These values are given as positive differnces
                      % from the one before the last bin number of negative
                      % differences from the one before the first bin numbers.
                      
    % The following variables are arrays of length n_blks, where n_blks is the
    % number of blocks used to generate these measurements.
    mean_msr = 0;     % mean of unquantized measurements
    stdv_msr = 0;     % standard deviation of unquantized measurements
    n_bins;           % number of bins in range (not including outlier bins)
    
    len_enc_hist = []; % No. of bytes used to encode histograms (empty = unknown)
    len_enc_msrs = []; % No. of bytes used to encode measurements (empty = unknown)
    
    % The following is an array of parameter vector as used by quantize(). 
    % It is not encoded and decoded
    params;
    
    intvl=[]; % Actual interval used for quantization. Empty means that it is 
              %   determined by the parameters.
  end
  
  properties (Access=private)
    saved_indcs = false;  % The logical value indicates that it is not set yet
  end
  
  methods
    function obj = QuantMeasurements(other)
      if (nargin == 1);
        obj = other.copy();
      end
    end
    
    function n_msrs = nMsrs(obj)
      n_msrs = length(obj.bin_no_clip) + length(obj.bin_numbers);
    end
    
    function n_sat_msrs = nSatMsrs(obj)
      % Returns the number of saturated measurements
      n_sat_msrs = length(find(obj.bin_numbers==obj.n_bins+1));
    end
    
    function n_lost_msrs = nLostMsrs(obj)
      % Returns the number of saturated and unsaved measurements
      n_lost_msrs = length(find(obj.bin_numbers > obj.n_bins)) - length(obj.saved);
    end
    
    function nbn = nBins(obj, iblk)
      if nargin < 2
        iblk = 1;
      end
      nbn = obj.n_bins(iblk);
    end
    
    function itvl = getIntvl(obj)
      if isempty(obj.intvl)
        itvl = [obj.params.intvl];
      else
        itvl = obj.intvl;
      end
    end
    
    function [entrp, use_prob, use_entrp] = compEntropy(obj)
      % Compute entropy of usable bin_numbrers. The usable bin_numbers are the 
      % entries of obj.bin_numbers which are <= obj.n_bins.
      % This function should be called only for single block objects
      % Input:
      %   obj - this object
      % Output:
      %   entrp - entropy of usable bin_numbrers, that is the expected number of
      %     nats needed to represent a used measurement
      %   use_prob - probability of that bin_numbers are usable, that is, <= obj.n_bins
      %   use_entrp -  expected number of bits used per measurement (taking into
      %     account that some measurements are dicarded)
      vals = double(obj.bin_numbers);
      if isempty(vals)
        entrp = 0;
        use_prob = 1;
        use_entrp = entrp;
      else
        used_vals = vals(vals<=obj.n_bins);
        n_vals = length(vals);
        use_prob = length(used_vals)/n_vals;
        hst = hist(used_vals, max(used_vals)-min(used_vals)+1);
        hst = hst(hst ~= 0)/n_vals;  % discard zeros and calc frequencies
        entrp = - dot(hst, log(hst));
        use_entrp = entrp/use_prob;
      end
    end
    
    function qmsr = getSingleBlk(obj, iblk)
      qmsr = QuantMeasurements();
      prms = obj.params(iblk);
      
      qmsr.params = prms;
      qmsr.params.clip_offset = zeros(1,1,'like',qmsr.params.clip_offset);
      qmsr.params.no_clip_offset = zeros(1,1,'like',qmsr.params.no_clip_offset);
      
      qmsr.bin_no_clip = ...
        obj.bin_no_clip(prms.no_clip_offset+1:prms.no_clip_offset+prms.n_no_clip);
      qmsr.bin_numbers = ...
        obj.bin_numbers(prms.clip_offset+1:prms.clip_offset+prms.n_clip);
      
      if isempty(obj.saved)
        qmsr.saved = [];
      else
        qmsr.saved = obj.saved(obj.saved_indcs>prms.clip_offset & ...
          obj.saved_indcs<=prms.clip_offset+prms.n_clip);
      end
      
      qmsr.mean_msr = obj.mean_msr(iblk);
      qmsr.stdv_msr = obj.stdv_msr(iblk);
      qmsr.n_bins = obj.n_bins(iblk);
      
      
    end
    
    function str = report(obj, info)
      % Generate a string with a report about the measurements. info is the
      % struct used for encode.
      
      str = sprintf('Measurements: %8d. Saturated: %5d (%.4f). Lost: %5d (%.4f)', ...
        obj.nMsrs(), obj.nSatMsrs(), obj.nSatMsrs()/obj.nMsrs(),...
        obj.nLostMsrs(), obj.nLostMsrs()/obj.nMsrs());
      
      str = sprintf('%s\n  Codewords: %s', str, show_str(obj.n_bins));
      bytes = obj.codeLength(info, true);
      bits = double(bytes)*8;
      str = sprintf('%s\n  Bytes: %8d. Bits/msr.: %6.3f b/msr. Bits/used msr.: %6.3f',...
        str, bytes, bits/obj.nMsrs(), bits/(obj.nMsrs()-obj.nLostMsrs()+1E-10));
    end
    
    function svd_ind = get.saved_indcs(obj)
      if islogical(obj.saved_indcs)
        obj.saved_indcs  = zeros(size(obj.saved));
        ofst = 0;
        for iblk=1:numel(obj.params)
          prms = obj.params(iblk);
          indcs = find(obj.bin_numbers(prms.clip_offset+1:...
            prms.clip_offset+prms.n_clip) == obj.n_bins(iblk)+1);
          obj.saved_indcs(ofst+1:ofst+length(indcs)) =indcs;
          ofst = ofst + length(indcs);
        end
      end
      svd_ind = obj.saved_indcs;
    end
    
    function l_hst = get.len_enc_hist(obj)
      if isempty(obj.len_enc_hist)
        obj.codeLength();  % Force encoding
      end
      l_hst = obj.len_enc_hist;
    end
    
    function l_msr = get.len_enc_msrs(obj)
      if isempty(obj.len_enc_msrs)
        obj.codeLength();  % Force encoding
      end
      l_msr = obj.len_enc_msrs;
    end    
    
    function len = encode(obj, code_dest, info)
      n_blks = length(obj.params);
      
      obj.len_enc_hist = 0;
      obj.len_enc_msrs = 0;
      
      mn_msr = obj.mean_msr(:);
      sd_msr = obj.stdv_msr(:);
      nbn = double(obj.n_bins);
      
      len_s = length(obj.saved);
      max_bin = obj.toCPUSInt((nbn-1)/2);
      
      % Write lengths (unsigned integers)
      len = code_dest.writeUInt(...
        [len_s; nbn(:)]);
      if ischar(len);  return; end
      
      % Write floating point parameters
      flt_vals = [mn_msr, sd_msr];
      n = code_dest.writeNumber(flt_vals);
      if ischar(n); len = n; return; end
      len = len + n;
      
      n = code_dest.writeNumber(obj.intvl);
      if ischar(n); len = n; return; end
      len = len + n;
      
      % Subtract max_bin and write no_clip bins (signed integers)
      bin_nc = obj.bin_no_clip;
      function sub_nc_bins(iblk)
        prms = obj.params(iblk);
        b = prms.no_clip_offset+1;
        e = prms.no_clip_offset+prms.n_no_clip;
        bin_nc(b:e) = bin_nc(b:e)  - max_bin(iblk);
      end
      arrayfun(@sub_nc_bins, 1:n_blks);
      
      if ~isempty(bin_nc)
        n = code_dest.writeSInt(bin_nc);
        if ischar(n); len = n; return; end
        len = len + n;
        obj.len_enc_msrs = obj.len_enc_msrs + n;
      end
      
      % Write saved bins (signed integers)
      n = code_dest.writeSInt(obj.saved);
      if ischar(n); len = n; return; end
      len = len + n;
      obj.len_enc_msrs = obj.len_enc_msrs + n;
      
      if info.enc_opts.lossless_coder == CS_EncParams.LLC_AC
        
        [md, acf, ar_bins] = arrayfun(@encodeBlkArith, 1:numel(nbn));
        
        % write md
        md = vertcat(md{:});
        if info.enc_opts.lossless_coder_AC_hist_type == ...
            CS_EncParams.LLC_AC_HST_FULL
          nb_hist_mode = 1;
        else
          nb_hist_mode = 2;
        end
        n = code_dest.writeBits(md, nb_hist_mode);
        if ischar(n); len = n; return; end
        len = len + n;
        obj.len_enc_hist = obj.len_enc_hist + n;
        
        % write acf
        acf = vertcat(acf{:});
        if length(acf) ~= length(md)
          error('length(acf) ~= length(md)');
        end
        for k=1:length(acf)
          n = acf{k}.encode(code_dest);
          if ischar(n); len = n; return; end
          len = len+n;
          obj.len_enc_hist = obj.len_enc_hist + n;
        end
        
        % write arlithmetic coding bits.
        arith_bins = vertcat(ar_bins{:});
        if length(arith_bins) ~= length(acf)
          error('length(arith_bins) ~= length(acf)');
        end
        for k=1:length(arith_bins)
          n = code_dest.writeBits(arith_bins{k});
          if ischar(n); len = n; return; end
          len = len + n;
          obj.len_enc_msrs = obj.len_enc_msrs + n;
        end
      else
        mxbn = max(max_bin);
        if mxbn < 128
          n_bits = nextpow2(mxbn)+1;
          n = code_dest.writeBitsArray(obj.bin_numbers, n_bits);
        elseif length(obj.params) == 1
          n = code_dest.writeSIntOffset(obj.bin_numbers, int16(max_bin));
        else
          vals = obj.bin_numbers;
          mx_bn = cast(max_bin, 'like', vals);
%          arrayfun(@sub_c_bins, (1:n_blks)', mx_bn);
          for ib = 1:n_blks
            sub_c_bins(ib, mx_bn(ib));
          end
          n = code_dest.writeSInt(vals);
        end
        if ischar(n); len = n; return; end
        len = len + n;
        obj.len_enc_msrs = obj.len_enc_msrs + n;
      end
        
      function sub_c_bins(iblk, mx)
        prms = obj.params(iblk);
        b = prms.clip_offset+1;
        e = prms.clip_offset+prms.n_clip;
        vals(b:e) = vals(b:e)  - mx;
      end
      
      function [md, acf, ar_bins] = encodeBlkArith(iblk)
        % encodeBlkArith - calculate arithmetic coding from one block
        %   Input argument:
        %     iblk - block number
        %   Output arguments
        %     use_gaus:  If true, Gaussian approximation was used for frequencies,
        %                else measurements histogram was used.
        %     hst: if use_gaus is false the histogram used to compute
        %          frequencies, else []. Wrapped by a single cell.
        %     ar_bins: The arithmetic doing bits, wrapped by a single cell
        
        ivl = obj.params(iblk).intvl;
        if isstruct(info.enc_opts) && ...
            ~isfield(info.enc_opts, 'lossless_coder_AC_gaus_thrsh')
          use_gaus = -1;
        elseif info.enc_opts.lossless_coder_AC_gaus_thrsh == -1
          use_gaus = -1;
        else
          use_gaus = ...
            info.quantizer.q_ampl_mltplr / ...
            info.quantizer.toCPUFloat(max_bin(iblk)) <...
            info.enc_opts.lossless_coder_AC_gaus_thrsh;
        end
        prms = obj.params(iblk);
        bin_nbr = obj.bin_numbers(prms.clip_offset+1: prms.clip_offset+prms.n_clip);
        [md, acf, nseq] = obj.calcFreq(use_gaus, ...
          info.enc_opts.lossless_coder_AC_hist_type, bin_nbr, ...
          sd_msr(iblk), nbn(iblk), ivl);
        ofset = cumsum([0, nseq(1:end-1)]);
        ofset = ofset(:)';
        
        ar_bins = arrayfun(@(ac,ofst,nsq) {ac{1}.encAC(bin_nbr(ofst+1:ofst+nsq)) },...
          acf, ofset, nseq);
        ar_bins = { ar_bins };
        acf = { acf };
        md = { md };
      end

    end
    
    function len = decode(obj, code_src, info, cnt)
      if nargin < 4
        cnt = inf;
      end
      
      obj.len_enc_hist = 0;
      obj.len_enc_msrs = 0;

      % Set params
      b_indx = info.vid_region.blk_indx;
      n_blks = size(b_indx,1);
      if info.enc_opts.par_blks > 0
        obj.params = info.quantizer.q_params(1: n_blks);
        obj.params = obj.params(:);
      else
        obj.params = info.quantizer.q_params(b_indx(1), b_indx(2),1);
      end
      
      % Read lengths (unsigned integers)
      [vals, nread] = code_src.readUInt(cnt, [n_blks+1, 1]);
      if ischar(vals) || (length(vals)==1 && vals == -1)
        len = vals;
        return
      end
      cnt = cnt - nread;
      len = nread;

      len_s = vals(1);
      nbn = vals(2:end);
      obj.n_bins = UniformQuantizer.toCPUIndex(nbn);
      max_bin = obj.toCPUSInt((obj.n_bins-1)/2);
      
      % read floating point values (mean, standard deviation, intvl...) 
      [valstat, nread] = code_src.readNumber(cnt, [n_blks,2]);
      if ischar(valstat)
        if isempty(vlstat)
          len = 'Unexpected end of Data';
        else
          len = valstat;
        end
        return
      end
      cnt = cnt - nread;
      len = len + nread;
      
      valstat = info.quantizer.toCPUFloat(valstat);
      obj.mean_msr = valstat(:,1);
      obj.stdv_msr = valstat(:,2);

      switch info.enc_opts.qntzr_wdth_mode 
        case {CS_EncParams.Q_WDTH_RNG, CS_EncParams.Q_WDTH_RNG_CSR}
          [q_intvl, nread] = code_src.readNumber(cnt, 1);
          if ischar(q_intvl); 
            if isempty(q_intvl)
              len = 'Unexpected end of Data';
            else
              len = q_intvl;
            end
            return
          end
          cnt = cnt - nread;
          len = len + nread;
          obj.intvl = info.quantizer.toCPUFloat(q_intvl);
      end

       % Read no_clip bins
      n_nc = obj.params(end).no_clip_offset + obj.params(end).n_no_clip;
      [vals1, nread] = code_src.readSInt(cnt, [n_nc, 1]);
      if ischar(vals1)
        if isempty(vals1)
          len = 'Unexpected end of Data';
        else
          len = vals1;
        end
        return
      end
      cnt = cnt - nread;
      len = len + nread;
      obj.len_enc_msrs = obj.len_enc_msrs + nread;
      
      vals1 = info.quantizer.toSInt(vals1);
      bin_nc = arrayfun(@add_no_clip, 1:n_blks);
      obj.bin_no_clip = vertcat(bin_nc{:});
      
      function bin_a = add_no_clip(iblk)
        prms = obj.params(iblk);
        bin_a  = { vals1(prms.no_clip_offset+1:...
          prms.no_clip_offset+prms.n_no_clip) + max_bin(iblk) };
      end
      
      %Read save values
      [vals, nread] = code_src.readSInt(cnt, [len_s, 1]);
      if ischar(vals)
        if isempty(vals)
          len = 'Unexpected end of Data';
        else
          len = vals;
        end
        return
      end
      cnt = cnt - nread;
      len = len + nread;
      obj.len_enc_msrs = obj.len_enc_msrs + nread;
      
      if isempty(vals)
        vals = zeros(0,1);
      end
      obj.saved = info.quantizer.toSInt(vals);
      
      if info.enc_opts.lossless_coder == CS_EncParams.LLC_AC
        % read md
        if info.enc_opts.lossless_coder_AC_hist_type == ...
            CS_EncParams.LLC_AC_HST_FULL
          nb_hist_mode = 1;
        else
          nb_hist_mode = 2;
        end
        [md, nread] = code_src.readBits(cnt, nb_hist_mode);
        if ischar(md); len = md; return; end
        cnt = cnt - nread;
        len = len + nread;
        obj.len_enc_hist = obj.len_enc_hist + nread;
        
        % read acf
        acf = cell(length(md),1);
        iblk = 1;
        nseq=0;
        for k=1:length(acf)
          [acf{k}, nread] = ACFreq.decode(code_src, ...
            md(k), obj.stdv_msr(iblk), nbn(iblk), ...
            obj.params(iblk).intvl, obj.params(iblk).n_clip, cnt);
          if ischar(acf{k}); len = acf{k}; return; end
          cnt = cnt - nread;
          len = len + nread;
          obj.len_enc_hist = obj.len_enc_hist + nread;
          nseq = nseq + acf{k}.nSeq();
          if nseq >= obj.params(iblk).n_clip
            nseq = nseq - obj.params(iblk).n_clip;
            iblk = iblk+1;
          end
        end
        
        % read arithmetic coding bits
        arith_bits = cell(size(acf));
        for k=1:numel(arith_bits)
          [arith_bits{k}, nread] = code_src.readBits(cnt);
          if ischar(arith_bits{k}); len = arith_bits{k}; return; end
          cnt = cnt - nread;
          len = len + nread;
          obj.len_enc_msrs = obj.len_enc_msrs + nread;
        end
        
        bnr = arrayfun(@(ac,acbits) ...
          {info.quantizer.toLabel(ac{1}.decAC(acbits{1}))}, ...
          acf,  arith_bits);
        
        obj.bin_numbers = vertcat(bnr{:});
      else
        % Read (signed integers)
        n_bin_numbers = obj.params(end).clip_offset + obj.params(end).n_clip;
        mxbn = max(max_bin);
        if mxbn < 128
          n_bits = nextpow2(mxbn)+1;
          [vals1, nread] = code_src.readBitsArray(n_bin_numbers, cnt, n_bits);
        else
          [vals1, nread] = code_src.readSInt(cnt, [n_bin_numbers, 1]);
        end
        if ischar(vals1)
          if isempty(vals1)
            len = 'Unexpected end of Data';
          else
            len = vals1;
          end
          return
        end
        len = len + nread;
        obj.len_enc_msrs = obj.len_enc_msrs + nread;
          
        if mxbn < 128
          obj.bin_numbers = info.quantizer.toLabel(vals1);
        else
          vals1 = info.quantizer.toSInt(vals1);
          b_vals = arrayfun(@(iblk) { vals1(obj.params(iblk).clip_offset+1:...
            obj.params(iblk).clip_offset+obj.params(iblk).n_clip) }, ...
            (1:n_blks)');
          b_vals = arrayfun(@(b,mx) {info.quantizer.toLabel(b{1} + mx)},...
            b_vals, max_bin);
          obj.bin_numbers = vertcat(b_vals{:});
        end
      end
    end
    
  end
  
  methods (Access=protected, Static)
    function [md, acf, nseq] = ...
        calcFreq(use_gaus, hist_type,bin_nbr, stdv, nbn, intvl)
      % calcFreq - calculate frequencies for arithmetic coding
      %   Input arguments:
      %     use_gaus: If 1, use Gaussian approximation for frequencies, 
      %               If 0, use actual histogram
      %               If -1, use the better of the previous two options.
      %     hist_type: One of CS_EncParams.LLC_AC_HST_xxxx
      %     bin_nbr: bin_numbers of clippable measurements
      %               
      %     stdv: Standard deviation of measurements;
      %     nbn: Number of bins
      %     intvl: quantization interval length
      %  Output arguments
      %     md - type of ACFreq
      %     acf - ACfreq used 
      
      if use_gaus == -1
        acf_g= ACFreqGaus(stdv, nbn, intvl, length(bin_nbr));
        [acf_h, h_est] = get_acf_hist();
        
        g_est = acf_g.estTotalLen(acf_h);
        if g_est < h_est
          acf = { acf_g };
        else
          acf = acf_h;
        end
      elseif use_gaus
        acf = { ACFreqGaus(stdv, nbn, intvl, length(bin_nbr)) };
      else
        acf = ACFreqHist.construct(nbn, bin_nbr); 
      end
      [md, nseq] = arrayfun(@get_md_nseq, acf);
      
      function [h_acf, h_len] = get_acf_hist()
        if hist_type == CS_EncParams.LLC_AC_HST_MLTI
          [h_acf, h_len] = ACFreqHist.construct(nbn, bin_nbr);
        else
          h_acf = ACFreqHist(nbn, bin_nbr);
          if hist_type == CS_EncParams.LLC_AC_HST_FULL
            h_acf.setMode(ACFreq.MODE_FULL_HIST);
          end
          h_len = h_acf.estTotalLen(); 
          h_acf = { h_acf };
        end
      end
      
      function [m,n] = get_md_nseq(ac)
        ac = ac{1};
        m = ac.getMode();
        n = ac.nSeq();
      end
    end
    
  end
end

