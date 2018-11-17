classdef SensingMatrixMD_DCT < SensingMatrixMD
  %SensingMatrixMD_DCT is a multi-dimensional DCT sensing matrix
  
  properties
  end
  
  methods
    function obj = SensingMatrixMD_DCT(opts)
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
      if nargin == 0
        return
      else
        obj.setSensingMatrixMD_DCT(opts);
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
      if nargin == 0
        return
      end
      obj.setSensingMatrixMD_DCT(opts);
    end
  end

  methods (Access = protected)
    function setSensingMatrixMD_DCT(obj, opts)
      %     opts: A struct with the following fields
      %       dims: array of dimension of each dimension, with the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements. Default: all possible
      %         measurements.
      
      opts.mtrcs = cell(size(opts.dims));
      for k=1:length(opts.mtrcs)
        opts.mtrcs{k} = struct('type', 'SensingMatrixBasicDCT', ...
          'args', struct('n_cols', opts.dims(k)));
      end
      obj.setSensingMatrixMD(opts);
    end
  end
end

