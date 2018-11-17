classdef ACFreqHist < ACFreq
  % ACFreqGaus ACFreq using histogram
  
  methods
    function obj = ACFreqHist(n_bins, seq)
      % Constructor. Can have one or two arguments
      % Input:
      %   seq - a sequence of labels
      
      obj = obj@ACFreq();
      
      if nargin > 0
        hst = histcounts(seq, 1:n_bins+2);
        obj.setFullHist(hst, length(seq))
      end
    end
    
    function n_lbl = nLabel(obj)
      % returns the number of different labels
      n_lbl = obj.n_lbl;
    end

    function ac_seq = encAC(obj, seq)
      seq_nz = obj.lbl_nz_inv(seq);
      ac_seq = obj.encAC@ACFreq(seq_nz);
    end

    function seq = decAC(obj, ac_seq)
      seq = obj.decAC@ACFreq(ac_seq);
      seq = obj.lbl_nz(seq);
      seq = cast(seq(:), 'int16');
    end

    function len = estACLen(obj, acf)
      % Estimate the arithmetically coded length (in bytes) of the ACFreq object acf,
      % assuming that the correct distribution is given in this object. if acf
      % is not given, assume that it is this object.
      if nargin < 2
        lgprb = log2(obj.prob);
      else
        [prb,nz] = acf.getProb();
        if length(intersect(nz, obj.lbl_nz)) < length(obj.lbl_nz)
          error('test ACFreq has zeros where refernce ACFreqHist does not')
        end
        lgprb = log2(prb(obj.lbl_nz));
      end
      len = ceil(-obj.nSeq() * dot(obj.prob, lgprb) / 8);
    end
    
    function md = getMode(obj)
      md = obj.mode;
    end
    
    function setMode(obj, md)
      len = sum(CodeStore.encodeLengthUInt(obj.hst_nz));
      switch md
        case ACFreq.MODE_FULL_HIST
          len = len + obj.n_lbl - length(obj.lbl_nz);
        case ACFreq.MODE_FLAG_HIST
          len = len + ceil(obj.n_lbl/8);
        case ACFreq.MODE_INDX_HIST
          len = len + CodeStore.encodeLengthBits(obj.lbl_nz, ceil(log2(obj.n_lbl)));
        otherwise
          error('Unexpected mode');
      end
      obj.hst_len = len;
      obj.mode = md;
    end
    
    function len = histLength(obj)
      len = obj.hst_len;
    end

    function len = encode(obj, code_dest)
      switch obj.mode
        case ACFreq.MODE_INDX_HIST
          hst = obj.hst_nz;
          cnt = code_dest.writeBits(obj.lbl_nz-1, nextpow2(obj.n_lbl));
        case ACFreq.MODE_FLAG_HIST
          hst = obj.hst_nz;
          cnt = code_dest.writeBitsArray(obj.lbl_nz_inv);
        case ACFreq.MODE_FULL_HIST
          hst = zeros(1,obj.nLabel(), 'like', obj.hst_nz);
          hst(obj.lbl_nz) = obj.hst_nz;
          cnt = 0;
      end
      if ischar(cnt)
        len = cnt;
        return
      end

      len = code_dest.writeUInt(hst);
      if ischar(len)
        return
      end
      len = len + cnt;
      
    end
    
    function cnt = decodeFullHist(obj, code_src, n_bins, max_cnt)
      if nargin < 4
        max_cnt = inf;
      end
      [hst, cnt] = code_src.readUInt(max_cnt, [1,n_bins+1]);
      if ischar(hst) || (isscalar(hst) && hst == -1)
        if ischar(hst)
          cnt = len;
        else
          cnt = 'EOD detected';
        end
        return
      end
      obj.setFullHist(hst, sum(hst));
    end
    
    function cnt = decodeFlagHist(obj, code_src, n_bins, max_cnt)
      if nargin < 4
        max_cnt = inf;
      end
      
      [nz, len] = code_src.readBitsArray(n_bins+1, max_cnt);
      if ischar(nz)
        cnt = nz;
        return
      end
      max_cnt = max_cnt - len;
      cnt = len;
      
      lb_nz = find(nz);
      
      [hst, len] = code_src.readUInt(max_cnt, [1,length(lb_nz)]);
      if ischar(hst) || (isscalar(hst) && hst == -1)
        if ischar(hst)
          cnt = len;
        else
          cnt = 'EOD detected';
        end
        return
      end
      cnt = cnt + len;
      obj.setIndxHist(hst, lb_nz, n_bins+1);
    end
    
    function cnt = decodeIndxHist(obj, code_src, n_bins, max_cnt)
      if nargin < 4
        max_cnt = inf;
      end
      n_bins = double(n_bins);
      [lb_nz, len] = code_src.readBits(max_cnt, nextpow2(n_bins+1));
      if ischar(lb_nz)
        cnt = lb_nz;
        return
      end
      cnt = len;
      max_cnt = max_cnt - len;
      lb_nz = lb_nz + 1;
      
      [hst, len] = code_src.readUInt(max_cnt, [1,length(lb_nz)]);
      if ischar(hst) || (isscalar(hst) && hst == -1)
        if ischar(hst)
          cnt = len;
        else
          cnt = 'EOD detected';
        end
        return
      end
      cnt = cnt + len;
      obj.setIndxHist(hst, lb_nz, n_bins+1);
    end
    
    function md = get.mode(obj)
      if isempty(obj.mode)
        obj.compMode();
      end
      md = obj.mode;
    end
    
    function len = get.hst_len(obj)
      if isempty(obj.hst_len)
        obj.compMode();
      end
      len = obj.hst_len;
    end
      
    function hni = get.lbl_nz_inv(obj)
      if isempty(obj.lbl_nz_inv)
        obj.lbl_nz_inv = zeros(1,obj.nLabel());
        obj.lbl_nz_inv(obj.lbl_nz) = (1:length(obj.lbl_nz));
      end
      hni = obj.lbl_nz_inv;
    end
  end
  
  methods(Static)
    function acf = combine(acf0, varargin)
      % Combines several ACFreqHist object, beginning with acf0, to create a new
      % ACFreqHist object acf;
      
      nsq = acf0.nSeq();
      hst = zeros(1,acf0.nLabel(), 'like', acf0.hst_nz);
      hst(acf0.lbl_nz) = acf0.hst_nz;
      
      for k=1:length(varargin)
        acf1 = varargin{k};
        if acf1.nLabel() ~= length(hst)
          error('different number of labels');
        end
        nsq = nsq + acf1.nSeq();
        hst(acf1.lbl_nz) = hst(acf1.lbl_nz) + acf1.hst_nz;
      end
      
      acf = ACFreqHist();
      acf.setFullHist(hst, nsq);
      
    end
    
    function [acfs, total_len] = construct(n_bins, seq)
      % Create several ACFreqHist objects to match a sequence of labels in a
      % locally optimal way.
      % Input:
      %   seq - label sequence
      %   n_bins - no. of non-saturated labels
      % Output:
      %   acfs - A cell array of ACFreqHist objects whose total nSeq() equals
      %          the length of seq
      %   total_len - the total of estTotalLen() of the elements of acfs
      
      % Create single label ACFreqHist objects (essentially run-length codes)
      seq = gather(seq(:)');
      
      h_end = find(seq(1:end-1) ~= seq(2:end));
      h_bgn = [1, h_end+1];
      h_end = [h_end, length(seq)];
      
      n_acf = length(h_bgn);
      acfs = cell(1,n_acf);
      acfs_len = zeros(size(acfs));
      for k=1:n_acf
        acfs{k} = ACFreqHist(n_bins, seq(h_bgn(k):h_end(k)));
        acfs_len(k) = acfs{k}.estTotalLen();
      end
      
      % Try to combine histgram in order to reduce total_len
      
      % Initialization. For each entry find the best combination of up to
      % MAX_STEP elements
      gain = zeros(size(acfs));
      step = zeros(size(acfs));
      acfn = acfs;
      acfn_len = acfs_len;
      for k=1:n_acf-1
        comp_gain(k);
      end
      [max_gain, max_indx] = max(gain);
      
      while max_gain > 0
        acfs{max_indx} = acfn{max_indx};
        stp = step(max_indx);
        acfs_len(max_indx) = acfn_len(max_indx);
        acfs(max_indx+1:max_indx+stp) = [];
        acfn(max_indx+1:max_indx+stp) = [];
        gain(max_indx+1:max_indx+stp) = [];
        step(max_indx+1:max_indx+stp) = [];
        acfs_len(max_indx+1:max_indx+stp) = [];
        acfn_len(max_indx+1:max_indx+stp) = [];
        n_acf = n_acf - stp;
        
        for k=max(1,max_indx-ACFreqHist.MAX_STEP):max_indx
          comp_gain(k);
        end
        [max_gain, max_indx] = max(gain);
      end
      
      total_len = sum(acfs_len);
      
      function comp_gain(k)
        acfg = acfs{k};
        old_len = acfs_len(k);
        gain(k) = 0;
        step(k) = 0;
        acfn_len(k) = acfs_len(k);
        for j=1:ACFreqHist.MAX_STEP
          if k+j > n_acf
            break;
          end
          acf = ACFreqHist.combine(acfs{k}, acfs{k+1:k+j});
          len = acf.estTotalLen();
          old_len = old_len + acfs_len(k+j);
          gn = old_len - len;
          if gn > gain(k)
            gain(k) = gather(gn);
            step(k) = j;
            acfg = acf;
            acfn_len(k) = gather(len);
          end
        end
        acfn{k} = acfg;
      end
    end
  end
  
  properties (Constant, Access=protected)
    MAX_STEP = 3;
  end
  
  properties (Access=protected)
    lbl_nz = [];   % Indices of labels with non-zero histogram bins
    hst_nz = [];   % Squeezed histogram
    lbl_nz_inv = []; % Array of size n_lbl, with pointers to the squeezed histogram.
    n_lbl = []; % no. of labels
    mode = [];
    hst_len = [];
  end
  
  methods (Access=protected)
    function setFullHist(obj, hst, nsq)
      % Set the object given the full histogram hst
      hst = gather(hst);
      obj.n_lbl = length(hst);
      obj.lbl_nz = find(hst);
      obj.hst_nz = hst(obj.lbl_nz);
      obj.set(gather(nsq));
    end
    
    function setIndxHist(obj, hst, lb_nz, n_lbl)
      obj.n_lbl = gather(n_lbl);
      obj.lbl_nz = gather(lb_nz);
      obj.hst_nz = gather(hst);
      nsq = sum(hst);
      obj.set(nsq);      
    end
    
    function set(obj, nsq)
      obj.set@ACFreq(obj.hst_nz, nsq);
    end
    
    function compMode(obj)
      len = ones(1,3) * sum(CodeStore.encodeLengthUInt(obj.hst_nz));
      len(ACFreq.MODE_FULL_HIST) = len(ACFreq.MODE_FULL_HIST) + ...
        obj.n_lbl - length(obj.lbl_nz);
      len(ACFreq.MODE_FLAG_HIST) = len(ACFreq.MODE_FLAG_HIST) + ceil(obj.n_lbl/8);
      len(ACFreq.MODE_INDX_HIST) = len(ACFreq.MODE_INDX_HIST) + ...
        CodeStore.encodeLengthBits(obj.lbl_nz, ceil(log2(obj.n_lbl)));
      [obj.hst_len, obj.mode] = min(len);
    end
    
    function [prb, nz] = getProb(obj)
      prb = zeros(1,obj.n_lbl);
      prb(obj.lbl_nz) = obj.prob;
      nz = obj.lbl_nz;
    end
    
  end
end