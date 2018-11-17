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
        
        % encode - a method which is abstract in CodeElement
        function len=encode(obj, code_dst, ~)
            len = code_dst.writeUInt(obj.n_rows, obj.is_trnsposed);
        end
        
        function len=decode(obj, code_src, ~, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            % Decode order
            [vals, n_read] = code_src.readUInt(cnt,[2,1]);
            if ischar(vals) || (isscalar(vals) && vals == -1)
                if ischar(vals)
                    len = vals;
                else
                    len = 'Unexpected end of data';
                end
                return;
            end
            len = n_read;
            ordr = vals(1);
            is_trnsp=vals(2);
            
            obj.setSensingMatrixUnit(ordr);
            if is_trnsp
                obj.transpose();
            end
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
        
        % returns the norm of A'A (transpose times the matrix the norm is
        % The norm is defined as max(obj.multTrnspVec(obj.multVec(x))) over 
        % all x such that x'x=1.
        function y = normAtA(~)
            y = 1;
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

