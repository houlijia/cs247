classdef SensingMatrixScaler < SensingMatrix
    %SensingMatrixScaler - A square matrix which mutliplies a vector by a scaler
    %   Detailed explanation goes here
    
    properties
        mltplr = 1;
    end
    
    methods
        % Constructor
        %   Input
        %     order - order of the square matrix
        %     mlt - multiplier (default = 1)
        function obj = SensingMatrixScaler(varargin)
            obj.setSensingMatrixScaler(varargin{:});
        end
        
        % Initialize the matrix
        %   Input
        %     obj - this object
        %     order - order of the square matrix
        %     mlt - multiplier (default = 1)
        function set(varargin)
            varargin = parseInitArgs(varargin, {'order', 'mlt'});
            obj.setSensingMatrixScaler(varargin{:});
        end
        
        % encode - a method which is abstract in CodeElement
        function len=encode(obj, code_dst, ~)
            len = code_dst.writeUInt(obj.n_rows, obj.is_trnsposed);
            if ischar(len)
                return;
            end
            cnt = len;

            len = code_dest.writeNumber(obj.mltplr);
            if ischar(len)
                return;
            end
            len = len+cnt;
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
            cnt = cnt - n_read;
            ordr = vals(1);
            is_trnsp=vals(2);
            
            [mlt, n_read] = code_src.readNumber(cnt);
            if ischar(mlt)
                if isempty(mlt)
                    mlt = 'Unexpected end of data';
                end
                len = mlt;
                return
            end
            len = len + n_read;
            
            obj.setSensingMatrixScaler(ordr, mlt);
            if is_trnsp
                obj.transpose();
            end
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = obj.mltplr * x;
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = obj.mltplr * x;
        end
        
        function y = doMultMat(obj,x)
          y = obj.mltplr * x;
        end
        
        function y = doMultTrnspMat(obj,x)
          y = obj.mltplr * x;
        end
        
        % Sometimes multTrnspVec may multiply the output y by a scaling
        % factor.  This function returns the scaling factor.
        function y = trnspScale(~)
            y = 1;
        end
        
        % returns the norm of A'A (transpose times the matrix the norm is
        % The norm is defined as max(obj.multTrnspVec(obj.multVec(x))) over 
        % all x such that x'x=1.
        function y = normAtA(obj)
            y = obj.mltplr * obj.mltplr;
        end
        
        function mtrx = compMatrix(obj)
          indcs = (1:obj.nRows())';
          mtrx = sparse(indcs, indcs, obj.scaler);
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          y = y / obj.mltplr;
        end

        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          y = y * obj.mltplr;
        end

        % Redefine getMatrix as getting a sparse matrix
        function mtrx = getMatrix(obj)
            if isempty(obj.matrix)
                obj.matrix = sparse(1:obj.n_rows,1:obj.n_rows, obj.mltplr,...
                    obj.n_rows, obj.n_cols);
            end
            mtrx = obj.matrix;
        end

        % returns true if getMatrix returns a sparse matrix
        function is_sprs = isSparse(~)
            is_sprs = true;
        end
        
    end
    
    methods (Access=protected)
        % Initialize the matrix
        %   Input
        %     obj - this object
        %     order - order of the square matrix
        %     mlt - multiplier (default = 1)
        function setSensingMatrixScaler(obj, order, mlt)
            if nargin < 2
                sm_args = {};
            else
                sm_args = {order, order};
            end
            obj.setSensingMatrix(sm_args{:});
            if nargin < 3
                obj.mltplr = 1;
            else
                obj.mltplr = mlt;
            end
        end
    end
    
end

