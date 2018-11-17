classdef SensingMatrixLclDCT < SensingMatrixSqrLclRnd & SensingMatrixDCT
  %SensingMatrixLclDCT Same as SensingMatrixDCT, except that the
  %randomization is local
  
  properties
  end
  
  methods
    function obj = SensingMatrixLclDCT(varargin)
      obj.set(varargin{:});
    end
    
    function set(obj, varargin)
      obj.set@SensingMatrixDCT(varargin{:});
      obj.setIndcsNoClip([], false);
    end
        
    function dc_val = getDC(obj,msrs)
      dc_val = obj.getDC@SensingMatrixSqrLclRnd(msrs);
    end
  end
  
  methods(Access=protected)
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqrLclRnd();
      obj.log2order = obj.toCPUIndex(obj.log2order);
    end
    
  end
  
end

