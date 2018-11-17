classdef SensingMatrixMatlab < SensingMatrix
    %SensingMatrixMatlab is an encapsulation of a Matlab matrix
    %   Detailed explanation goes here
    
    properties(Access=protected)
        mtrx_nrm = []
    end
    
    methods
        function obj=SensingMatrixMatlab(varargin)
            varargin = parseInitArgs(varargin, {'mtrx'});
            obj.setSensingMatrixMatlab(varargin{:});
        end
        
        function set(obj, varargin)
            obj.setSensingMatrixMatlab(varargin{:});
        end
        
        % encode - a method which is abstract in CodeElement
        function len=encode(obj, code_dst, ~)
            len = obj.encode@SensingMatrix(code_dst);
            if ischar(len)
                return;
            end
            total = len;
            
            issp = issparse(obj.matrix);
            len = code_dst.writeUInt(issp);
            if ischar(len)
                return;
            end
            total = total+len;
            
            if issp
                [cl,rw,vl] = find(obj.matrix);
                len = code_dst.writeUInt(length(cl));
                if ischar(len)
                    return;
                end
                total = total+len;
                
                len = code_dst.writeUInt([cl,rw]);
                if ischar(len)
                    return;
                end
                total = total+len;
                
                len = code_dst.writeNumber(vl);
                if ischar(len)
                    return;
                end
                total = total+len;
            else
                len = code_dst.writeNumber(obj.matrix);
                if ischar(len)
                    return;
                end
                total = total+len;
            end
            len = total;        
        end

        % decode - a method which is abstract in CodeElement
        function len=decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            len = obj.decode@SensingMatrix(code_src, info, cnt);
            if ischar(vals) || (isscalar(vals) && vals==-1)
                len = vals;
                return;
            end
            total = len;
            cnt = cnt-len;
            
            [issp, len] = code_src.readUInt(cnt);
            if ischar(issp) 
                len = issp;
                return;
            elseif isscalar(issp) && issp == -1
                len = 'unexpected EOD';
                return;
            end
            total = total+len;
            cnt = cnt-len;
            
            if issp
                [n_nz, len] = code_src.readUInt(cnt);
                if ischar(n_nz)
                    len = issp;
                    return;
                elseif isscalar(issp) && issp == -1
                    len = 'unexpected EOD';
                    return;
                end
                total = total+len;
                cnt = cnt-len;
                
                [ivl, len] = code_src.readUInt(cnt, [n_nz,2]);
                if ischar(ivl)
                    len = ivl;
                    return;
                elseif isscalar(ivl) && ivl == -1
                    len = 'unexpected EOD';
                    return;
                end
                total = total+len;
                cnt = cnt-len;
                
                [vl, len] = code_src.readNumber(cnt, [n_nz,1]);
                if ischar(vl)
                    len = vl;
                    if isempty(vl)
                        len = 'unexpected EOD';
                    end
                    return;
                end
                total = total+len;
                
                obj.setMatrix(sparse(ivl(:,1),ivl(:,2),vl,obj.nRows(),obj.nCols()));
            else
               [mtrx, len] = code_src.readNumber(cnt, [obj.nRows(), obj.nCols()]);
                if ischar(mtrx)
                    len = mtrx;
                    if isempty(mtrx)
                        len = 'unexpected EOD';
                    end
                    return;
                end
                total = total+len;
                
                obj.setMatrix(mtrx);
            end
                
            len = total;
        end

        % returns true if getMatrix returns a sparse matrix
        function is_sprs = isSparse(obj)
            is_sprs = issparse(obj.matrix);
        end
        
        function y = doMultVec(obj, x)
            y = obj.matrix * x;
        end
        
        function y = doMultTrnspVec(obj, x)
            y = obj.matrix' * x;
        end

        function y = doMultMat(obj, x)
            y = obj.matrix * x;
        end
        
        function y = doMultTrnspMat(obj, x)
            y = obj.matrix' * x;
        end

        function y = trnspScale(~)
            y = 1;
        end
        
        function y = normAtA(obj)
            if isempty(obj.mtrx_nrm)
                obj.mtrx_nrm = max(sum(abs(obj.matrix),2));
            end
            y = obj.mtrx_nrm;
        end
        
        function y = compMsrsNormalizer(obj)
          mtx = obj.matrix;
          if ~obj.is_transposed
            mtx = mtx';
          end
          y = sqrt(sum(mtx .* mtx))';
        end
          
        
    end
    
    methods (Access=protected)
       function setSensingMatrixMatlab(obj, mtrx)
            if nargin >1
                sm_args = {size(mtrx,1), size(mtrx,2)};
            else
                sm_args = {};
            end
            
            obj.setSensingMatrix(sm_args{:});
            
            if nargin > 1
                obj.matrix = mtrx;
            end
       end
        
    end
    
end

