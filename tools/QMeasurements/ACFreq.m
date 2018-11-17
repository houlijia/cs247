classdef (Abstract) ACFreq < handle
  properties (Constant)
    MODE_GAUS = 0; % Using Gaussian approximation
    MODE_FULL_HIST = 1; % Histogram with entries for each bin
    MODE_FLAG_HIST = 2; % Histogram with a flags for non-zero entries
    MODE_INDX_HIST = 3; % Histogram with an index table of non-zero entries
  end
  
  methods
    function obj = ACFreq(frq, nsq)
      % Constructor
      % Input(optional)
      %   frq: Frequencies values
      %   nsq: length of sequence
      if nargin > 0
        if nargin > 1
          obj.set(frq, nsq);
        else
          obj.set(frq);
        end
      end
    end
    
    function nsq = nSeq(obj)
      % returns number of elements in the labels sequence
      nsq = obj.n_seq;
    end
    
    function n_lbl = nLabel(obj)
      % returns the number of different labels
      n_lbl = length(obj.freq);
    end

    function ac_seq = encAC(obj,seq)
      % Encode the label sequence seq using arithmetic coding and return the
      % result in ac_seq, which is a uint8 seqeunce. seq is assumed to be
      % of length obj.nSeq()
      if length(obj.freq) > 1
        ac_seq = arithenco(gather(double(seq)), obj.freq);
      else
        ac_seq = [];
      end
      ac_seq = uint8(ac_seq);
    end
    
    function seq = decAC(obj, ac_seq)
      % Decode n_seq lables using arithmetic coding from the code sequence
      % ac_seq and return the labels in seq
      if length(obj.freq) > 1
        seq = arithdeco(double(ac_seq(:)), obj.freq, obj.nSeq());
      else
        seq = ones(obj.nSeq(),1);
      end      
    end
    
    function len = estTotalLen(obj, refs)
      % Estimates the total length for coding this object (histogram plus
      % arithmentic coding of data) assuming that the true distribution is given
      % by refs. refs can be a single ACFreqHist object or a cell array of such
      % objects, and the total of nSeq() of refs should equal nSeq() of obj.
      %
      % If obj is a ACFreqHist object the second argument may not be specified,
      % in which case the reference is assumed to be the object itself.
      
      len = obj.histLength() + 0.25; % Add 2 bits for mode
      if nargin < 2
        len = len + obj.estACLen();
        return
      elseif ~iscell(refs)
        refs = {refs};
      end
      
      nsq = 0;
      for k=1:numel(refs)
        len = len + refs{k}.estACLen(obj);
        nsq = nsq + refs{k}.nSeq();
      end
      if nsq ~= obj.nSeq()
        error('nSeq() do not add up');
      end
    end
    
    function prb = get.prob(obj)
      if isempty(obj.prob)
        obj.prob = obj.freq / sum(obj.freq);
      end
      prb = obj.prob;
    end
    
  end
  
  methods (Abstract)
    len = encode(obj, code_dest);  % Encode the parameters of the object
    md = getMode(obj); % return the mode of the histogram
    len = histLength(obj);  % Estimated coded length of histogram (in bytes)
  end
  
  methods(Static)
    function [acf, cnt] = ...
        decode(code_src, md, stdv_msr, n_bins, intvl, n_seq_blk, max_cnt)
      % Read and parse an ACFreq object
      % Input:
      %   code_src - CodeSrc object to read from
      %   md - Mode of object (one of ACFreq.MODE_...)
      %   n_bins - number of bins in range (not including outliers)
      %   n_seq_blk - length of the sequence for the whole block
      %   max_cnt - (optional) maximum number of bytes to read. Default=inf
      % Output
      %   acf - the acf object which was read, or an error string
      %   cnt - number of bytes which were read
      if nargin < 7
        max_cnt = inf;
      end
      cnt = 0;
      if md == ACFreq.MODE_GAUS
        acf = ACFreqGaus(stdv_msr, n_bins, intvl, n_seq_blk);
      else
        acf = ACFreqHist();
        
        switch md
          case ACFreq.MODE_FULL_HIST
            cnt = acf.decodeFullHist(code_src, n_bins, max_cnt);
          case ACFreq.MODE_FLAG_HIST
            cnt = acf.decodeFlagHist(code_src, n_bins, max_cnt);
          case ACFreq.MODE_INDX_HIST
            cnt = acf.decodeIndxHist(code_src, n_bins, max_cnt);
          otherwise
            error('Unrecognized mode: %s', show_str(md));
        end
      end
    end
  end
    
  
  properties (Access=protected)
    freq; % Array of label frequencies
    prob=[]; % probability corresponding to frequency
    est_len = []; % estimated length
    n_seq;
  end
  
  methods(Access=protected)
    function set(obj, frq, nsq)
      % Perform constructor operation
      %   obj - this object
      %   frq - Array for freq
      %   nsq - length of sequence 
      obj.freq = gather(double(frq));
      if nargin > 2
        obj.n_seq = nsq;
      end
    end
    
    function [prb, nz] = getProb(obj)
      % Return the unsqueezed probability and the indices of non-zero
      % probability.
      prb = obj.prob;
      nz = 1:obj.nLabel();
    end
  end
  
  
end
