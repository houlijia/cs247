classdef SensingMatrixTrnsfrm < SensingMatrix
  %SensingMatrixTrnsfrm preform a Transform. It is assumed that the
  %transform is orthogonal with norm 1.
  
  properties
    drct = [];
    invr = [];
  end
  
  methods
    % Constructor
    %   Input:
    %     order - Transform order.
    %     trnsfrm - Direct transform function
    %     inv_trnsfrm
    function obj = SensingMatrixTrnsfrm(varargin)
      obj.setSensingMatrixTrnsfrm(varargin{:})
    end
    
    % set initialize
    %   Input:
    %     obj - this object
    %     order - DCT order.
    %     trnsfrm - Direct transform function
    %     inv_trnsfrm
    function set(obj, varargin)
      varargin = parseInitArgs(varargin, {'order', 'trnsfrm', 'inv_trnsfrm'});
      obj.setSensingMatrixTrnsfrm(varargin{:})
    end
    
    function y = doMultVec(obj,x)
      y = obj.drct(x);
    end
    
    function y = doMultTrnspVec(obj,x)
      y = obj.invr(x);
    end
    
    function y = doMultMat(obj, x)
      y = obj.drct(x);
    end
    
    function y = doMultTrnspMat(obj, x)
      y = obj.invr(x);
    end
    
  end
  
  methods (Access=protected)
    function setSensingMatrixTrnsfrm(obj, order, trnsfrm, inv_trnsfrm)
      if nargin > 1
        obj.setSensingMatrix(order, order);
        if nargin > 2
          obj.drct = trnsfrm;
          if nargin > 3
            obj.invr = inv_trnsfrm;
          end
        end
      end
      obj.setOrtho_col(true);
      obj.nrm = obj.toCPUFloat(1);
      obj.nrm_inv = obj.toCPUFloat(1);
      obj.exct_nrm = [true, true];
      
    end
  end

  
end

