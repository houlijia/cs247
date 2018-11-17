classdef SensingMatrixSqrLclRnd <  SensingMatrixSqrLcl & SensingMatrixSqrRnd
  %SensingMatrixSqrLclRnd is the same as SensingMatrixSqrRnd, except that
  %instead of SensingMatrixSqr we use SensingMatrixSqrLcl
  
  properties
  end
  
  methods
    %Constructor
    function obj = SensingMatrixSqrLclRnd(varargin)
      obj.set(varargin{:});
    end
    
    function set(obj, varargin)
      obj.set@SensingMatrixSqrRnd(varargin{:})
    end
    
    function y=doMultVec(obj, varargin)
      y = obj.doMultVec@SensingMatrixSqrLcl(varargin{:});
    end
    
    function y=doMultTrnspVec(obj,varargin)
      y = obj.doMultTrnspVec@SensingMatrixSqrLcl(varargin{:});
    end
    
    function y=doMultMat(obj, varargin)
      y = obj.doMultMat@SensingMatrixSqrLcl(varargin{:});
    end
    
    function y=doMultTrnspMat(obj,varargin)
      y = obj.doMultTrnspMat@SensingMatrixSqrLcl(varargin{:});
    end
  end
  
  methods(Access=protected)

    function [PL, PR] = makePermutations(obj, sqr_order, opts)
      PL = obj.makeRowSlctPermutation(sqr_order, opts);
      if obj.unit_permut_R
        PR = [];
      else
        PR = obj.makeRandomizerPermutation(obj.n_cols);
      end
    end
    
   function PL = makeRowSlctPermutation(obj,order,opts)
      PL = obj.rnd_strm.randperm(order, opts.n_rows)';
   end
   
   function PR = makeRandomizerPermutation(obj,order)
      % Randomly select indices which need a sign change
      PR = obj.toIndex(find(obj.rnd_strm.randi([0,1],[1,order]))); %#ok
   end
   
   function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqrLcl();
    end
  end
  
end

