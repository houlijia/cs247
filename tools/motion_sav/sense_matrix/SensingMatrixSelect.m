classdef SensingMatrixSelect < SensingMatrix
    %SensingMatrixSelect is a matrix where each row is all zeros except for
    %one entry of 1.  Essentially multiplying by the matrix retruns a
    %a selection of the entries of the original vector.
    
    properties
        slct_indices=[];
        trnsp_fctr = 1;
    end
    
    methods
        % constructor can be called with either 0, 2 or 3 arguments.
        % Input:
        %   indcs - an array of indices selected by the matrix (its length
        %           is the number of rows)
        %   num_columns - number of columns
        %   tr_fctr - (optional) transpose factor
        function obj = SensingMatrixSelect(varargin)
            obj.setSensingMatrixSelect(varargin{:})
        end
        
        % Initialize the matrix
        % Input:
        %   obj - this object
        %   indcs - an array of indices selected by the matrix (its length
        %           is the number of rows)
        %   num_columns - number of columns
        %   tr_fctr - (optional) transpose factor
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'indcs', 'num_columns', ...
                'tr_fctr'});
            obj.setSensingMatrixSelect(varargin{:})
        end
        
        function setTrnspFctr(obj, tr_fctr)
            obj.trnsp_fctr = tr_fctr;
        end
        
        function len=encode(obj, code_dst, ~)
            len = obj.encode@SensingMatrix(code_dst);
            if ischar(len)
                return
            end
            
            len0 = code_dst.writeUInt(obj.slct_indices);
            if ischar(len0)
                len = len0;
                return
            end
            len = len + len0;
        end

        function len=decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            len = obj.decode@SensingMatrix(code_src, info, cnt);
            if ischar(len) || (isscalar(len) && len==-1)
                return;
            end
            
            [indcs, len0] = code_src.readUInt(cnt-len, [obj.n_rows, 1]);
            if ischar(indcs)
                len = indcs;
                return;
            elseif isscalar(indcs) && indcs < 0
                len = 'EOD encountered';
                return
            end
            len = len+len0;
            obj.slct_indices = double(indcs);
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = x(obj.slct_indices);
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return a vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = zeros(obj.n_cols, 1);
            y(obj.slct_indices) = x * obj.trnsp_fctr;
        end
        
        % Sometimes multTrnspVec may multiply the output y by a scaling
        % factor.  This function returns the scaling factor.
        function y = trnspScale(obj)
            y = obj.trnsp_fctr;
        end
        
        function y = doMultMat(obj, x)
            y = x(obj.slct_indices,:);
        end
        
        function y = doMultTrnspMat(obj, x)
            y = zeros(obj.n_cols, size(x,2));
            y(obj.slct_indices,:) = x * obj.trnsp_fctr;
        end
        
          % returns the norm of A'A (transpose times the matrix the norm is
        % The norm is defined as max(obj.multTrnspVec(obj.multVec(x))) over 
        % all x such that x'x=1.
        function y = normAtA(~)
            y = 1;
        end
        
        % Redefine getMtrx as getting a sparse matrix
        function mtrx = compMatrix(obj)
          mtrx = sparse(1:length(obj.slct_indices),...
            obj.slct_indices, 1, obj.n_rows, obj.n_cols);
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
        
        % returns true if getMatrix returns a sparse matrix
        function is_sprs = isSparse(~)
            is_sprs = true;
        end
    end
 
    methods (Access=protected)
        % Initialize the matrix
        % Input:
        %   obj - this object
        %   indcs - an array of indices selected by the matrix (its length
        %           is the number of rows)
        %   num_columns - number of columns
        %   tr_fctr - (optional) transpose factor
        function setSensingMatrixSelect(obj, indcs, num_columns, tr_fctr)
            if nargin <= 1
                sm_args  = {};
            else
                sm_args = {length(indcs), num_columns};
            end
            
            obj.setSensingMatrix(sm_args{:});
            
            if nargin > 1
                if ~iscolumn(indcs)
                    indcs = indcs';
                end
                obj.slct_indices = indcs;
                if nargin >=4
                    obj.setTrnspFctr(tr_fctr);
                end
            end
        end
    end    
end

