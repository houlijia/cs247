classdef SensingMatrixSqrNrRnd <  SensingMatrixSqrRnd
  %SensingMatrixSqrNrRnd is the same as SensingMatrixSqrRnd, except that
  %the input is not reordered (PR is the unit permutation).
  
  properties
  end
  
  methods
    %Constructor
    function obj = SensingMatrixSqrNrRnd(varargin)
      obj.set(varargin{:});
    end
    
    function set(obj, varargin)
      obj.set@SensingMatrixNrRnd(varargin{:})
    end
    
    function y=doMultVec(obj, x)
      % doMultVec - implemenentation of abstract method of
      % SensingMatrix - multiply a vector x by A.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      u = [x; zeros(obj.sqr_order-size(x,1),1, 'like', x)];
      v =obj.mltSqr(u);
      if ~isempty(obj.PL)
        y = v(obj.PL(:),1);
      else
        y = v(1:obj.n_rows,1);
      end
    end
    
    function y=doMultTrnspVec(obj,x)
      % doMultTrnspVec - implemenentation of abstract method of
      % SensingMatrix - multiply a vector x by A'.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      u = zeros(obj.sqr_order, 1, 'like', x);
      if ~isempty(obj.PL)
        u(obj.PL(:),1) = x;
      else
        u(1:obj.n_rows,1) = x;
      end
      v =obj.mltTrnspSqr(u);
      y = v(1:obj.n_cols);
    end
    
    function y=doMultMat(obj, x)
      % doMultVec - implemenentation of abstract method of
      % SensingMatrix - multiply a vector x by A.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      u = [x; zeros(obj.sqr_order-size(x,1), size(x,2), 'like', x)];
      v =obj.mltSqr(u);
      if ~isempty(obj.PL)
        y = v(obj.PL(:),:);
      else
        y = v(1:obj.n_rows,:);
      end
    end
    
    function y=doMultTrnspMat(obj,x)
      % doMultTrnspVec - implemenentation of abstract method of
      % SensingMatrix - multiply a vector x by A'.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      u = zeros(obj.sqr_order, size(x,2), 'like', x);
      if ~isempty(obj.PL)
        u(obj.PL(:),:) = x;
      else
        u(1:obj.n_rows,:) = x;
      end
      v = obj.mltTrnspSqr(u);
      y = v(1:obj.n_cols,:);      
    end
  end
  
  methods(Access=protected)
    function [PL, PR] = makePermutations(obj, order, opts)
      PL = obj.makeRowSlctPermutation(order, opts);
      PR = [];
    end
    
   function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqrRnd();
   end
    
  end
  
end
