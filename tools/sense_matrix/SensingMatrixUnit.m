classdef SensingMatrixUnit < SensingMatrixScaler
  %SensingMatrixUnit - A unit matrix
  
  properties
  end
  
  methods
        % Constructor
        %   Input
        %     order - order of the square matrix
        function obj = SensingMatrixUnit(varargin)
            obj.setSensingMatrixUnit(varargin{:});
        end
        
        % Initialize the matrix
        %   Input
        %     obj - this object
        %     order - order of the square matrix
        %     mlt - multiplier (default = 1)
        function set(varargin)
            varargin = parseInitArgs(varargin, {'order'});
            obj.setSensingMatrixUnit(varargin{:});
        end
        
        %change the matrix to be its inverse
        function invert(~)
        end
    
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(~, x)
            y = x;
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(~, x)
            y = x;
        end
        
        function y = doMultMat(~,x)
          y = x;
        end
        
        function y = doMultTrnspMat(~,x)
          y = x;
        end
        
        % Multiply a matrix or a vector by the matrix whose entries are the
        % absolute values of the entries of this matrix.
        function y = multAbs(~,x)
          y = x;
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(~,y)
        end

        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(~,y)
        end

  end
  
  methods (Access=protected)
    % Initialize the matrix
    %   Input
    %     obj - this object
    %     order - order of the square matrix
    function setSensingMatrixUnit(obj, order)
      if nargin < 2
        sm_args = {};
      else
        sm_args = {order, 1};
      end
      obj.setSensingMatrixScaler(sm_args{:});
    end
  end
end

