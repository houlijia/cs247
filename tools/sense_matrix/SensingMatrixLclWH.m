classdef SensingMatrixLclWH < SensingMatrixSqrLclRnd & SensingMatrixWH
  %SensingMatrixLclWH Same as SensingMatrixWH, except that the
  %randomization is local
  
  properties
  end
  
  methods
    function obj = SensingMatrixLclWH(varargin)
        % Set Initialize
        %   Input:
        %     obj - this object
        %     n_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     n_cols - number of columns
        %     rnd_seed - random number generation seed
        %     prmt_info - a struct containing parameters for premtutation and selection
        %              the only relevant field is PL_mode.      %
        %     wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
        %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        %     nodc - Do not include the DC measurement in the measurements vector
       obj.set(varargin{:})
    end
    
    function set(obj, varargin)
        % Set Initialize
        %   Input:
        %     obj - this object
        %     n_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     n_cols - number of columns
        %     rnd_seed - random number generation seed
        %     prmt_info - a struct containing parameters for premtutation and selection
        %              the only relevant field is PL_mode.      %
        %     wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
        %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        %     nodc - Do not include the DC measurement in the measurements vector
       obj.set@SensingMatrixWH(varargin{:});
       obj.setIndcsNoClip([], false);
    end
    
    function md = whMode(obj)
      md = obj.whMode@SensingMatrixWH();
    end
    
    function md_name = whModeName(obj)
      md_name = obj.whModeName@SensingMatrixWH();
    end
    
    function dc_val = getDC(obj,msrs)
      dc_val = obj.getDC@SensingMatrixSqrLclRnd(msrs);
    end
  end
  
  methods(Access=protected)
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqrLclRnd();
      obj.setCastIndex@SensingMatrixWH();
    end
    
    function PL = makeRowSlctPermutation(obj,order, opts)
      prmt = opts.prmt;
      if prmt.PL_mode == SensingMatrixWH.SLCT_MODE_ARBT
        if ~isfield(opts, 'nodc') || ~opts.nodc
          PL = obj.rnd_strm.randperm(order, obj.n_rows)';
        elseif obj.n_rows >= order
          error('cannot have n_rows >= order with nodc');
        else
          PL = 1+obj.rnd_strm.randperm(order-1, obj.n_rows)';
        end
      else
        PL = obj.makeRowSlctPermutation@SensingMatrixWH(order, opts);
      end
    end
  end
  
end

