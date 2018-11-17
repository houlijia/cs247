classdef SensingMatrixSqrLcl < SensingMatrixSqr
  % SensingMatrixSqrLcl is the same as SensingMatrixSqr, except that
  % Pright is not a permutation matrix but a
  % diagonal matrix in which the diagonal elements get values of +-1.
  
  properties
  end
  
  
  methods
    % Constructor. Can have 1, 2 or 5 arguments or none
    function obj = SensingMatrixSqrLcl(varargin)
      obj.setSensingMatrixSqr(varargin{:});
    end
    
    function set(obj, varargin)
      varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
        'order', 'PL', 'PR'});
      obj.setSensingMatrixSqr(varargin{:});
    end
    
    function y=doMultVec(obj, x)
      % doMultVec - implemenentation of abstract method of SensingMatrix -
      % multiply a vector x by A. The implementation uses the abstract
      % function multSqr().
      % INPUT
      %    obj - this object
      %    x   - input vector.
      
      x(obj.PR) = -x(obj.PR);
      x = [x; zeros(obj.sqr_order-size(x,1), 1, 'like', x)];
      v = obj.mltSqr(x);
      if ~isempty(obj.PL)
        y = v(obj.PL);
      else
        y = v(1:obj.n_rows);
      end
    end
    
    function y=doMultTrnspVec(obj,x)
      % doMultTrnspVec - implemenentation of abstract method of SensingMatrix -
      % multiply a vector x by A'.
      % INPUT
      %    obj - this object
      %    x   - input vector.
      
      u = zeros(obj.sqr_order, 1, 'like', x);
      if ~isempty(obj.PL)
        u(obj.PL) = x;
      else
        u(1:obj.n_rows) = x;
      end
      v = obj.mltTrnspSqr(u);
      y = v(1:obj.n_cols);
      y(obj.PR) = -y(obj.PR);
    end
    
    function y=doMultMat(obj, x)
      x(obj.PR,:) = -x(obj.PR,:);
      x = [x; zeros(obj.sqr_order-size(x,1), size(x,2), 'like', x)];
      v = obj.mltSqr(x);
      if ~isempty(obj.PL)
        y = v(obj.PL,:);
      else
        y = v(1:obj.n_rows,:);
      end
    end
    
    function y=doMultTrnspMat(obj,x)
      u = zeros(obj.sqr_order, size(x,2), 'like', x);
      if ~isempty(obj.PL)
        u(obj.PL,:) = x;
      else
        u(1:obj.n_rows,:) = x;
      end
      v = obj.mltTrnspSqr(u);
      y = v(1:obj.n_cols,:);
      y(obj.PR,:) = -y(obj.PR,:);
      
    end
    
    function setPermutations(obj, order, PL, PR)
      obj.setPermutations@SensingMatrixSqr(order, PL, PR);
    end
    
    % Returns the sum of values of the measurement which contain DC value,
    % weighted by the ratio of the DC value to other components (in
    % terms of RMS), or 0 if there is no such measurement.
    %   Input:
    %     obj - this object
    %     msrs - the measurements vector
    function dc_val = getDC(~,~)
      dc_val = 0;
    end
  end
  
  methods(Access=protected)
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqr();
    end
  end
  
end
