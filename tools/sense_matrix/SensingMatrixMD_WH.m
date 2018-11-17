classdef SensingMatrixMD_WH < SensingMatrixMD
  %SensingMatrixMD_WH is a multi-dimensional WH sensing matrix
  
  properties
  end
  
  methods
    function obj = SensingMatrixMD_WH(opts)
      % Constructor
      %   Input:
      %     opts: A struct with the following fields
      %       dims: array of dimension of each dimension, with the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements. Default: all possible
      %         measurements.
      %       wh_mode: (optional) 1,2,3 or 'hadamard','dyadic','sequency'.
      %         Default:'sequency'
      if nargin == 0
        return
      else
        obj.setSensingMatrixMD_WH(opts);
      end
    end
    
    function set(obj, opts)
      %   Input:
      %     obj: This object
      %     opts: A struct with the following fields
      %       dims: array of dimension of each dimension, with the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements. Default: all possible
      %         measurements.
      %       wh_mode: (optional) 1,2,3 or 'hadamard','dyadic','sequency'.
      %         Default:'sequency'
      if nargin == 0
        return
      end
      obj.setSensingMatrixMD_WH(opts);
    end
  end

  methods (Access = protected)
    function setSensingMatrixMD_WH(obj, opts)
      %     opts: A struct with the following fields
      %       dims: array of dimension of each dimension, with the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements. Default: all possible
      %         measurements.
      %       wh_mode: (optional) 1,2,3 or 'hadamard','dyadic','sequency'.
      %         Default:'sequency'
      
      if ~isfield(opts, 'wh_md')
        opts.wh_md = 'sequency';
      end
      opts.mtrcs = cell(size(opts.dims));
      for k=1:length(opts.mtrcs)
        opts.mtrcs{k} = struct('type', 'SensingMatrixBasicWH', ...
          'args', struct('order', opts.dims(k), 'wh_md', opts.wh_md));
      end
      obj.setSensingMatrixMD(opts);
    end
  end
end

