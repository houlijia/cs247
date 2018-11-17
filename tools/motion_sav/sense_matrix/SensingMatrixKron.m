classdef SensingMatrixKron < SensingMatrixComposed
  % SensingMatrixKron is a Kronecker product of several matrices:
  % mtrx{1} .*. mtrx{2} .*. ... where .*. denotes Kronecker product
   properties
     dims; % dimension of matrics, in reverse order of matrices
   end
    
    methods
        % Constructor
        %   Input:
        %     mtrcs - the matrices sequence
       function obj = SensingMatrixKron(varargin)
            obj.setSensingMatrixKron(varargin{:})
        end
        
        % Initialize the object
        %   Input:
        %     obj - this object
        %     mtrcs - the matrices sequence
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'mtrcs'});
            obj.setSensingMatrixKron(varargin{:})
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
          y = obj.doMult(x, obj.dims);
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
          dms = obj.dims;
          dms(:,[2 1]) = dms;
          for k=1:length(obj.mtrx)
            obj.mtrx{k}.transpose();
          end
          y = obj.doMult(x, dms);
          for k=1:length(obj.mtrx)
            obj.mtrx{k}.transpose();
          end
        end
        
        function y = doMultMat(obj, x)
          nc = size(x,2);
          dms = [obj.dims; nc, nc];
          y = doMult(x, dms);
        end
          
        function y = doMultTrnspMat(obj, x)
          nc = size(x,2);
          dms = [obj.dims; nc, nc];
          dms(:,[2 1]) = dms;
          for k=1:length(obj.mtrx)
            obj.mtrx{k}.transpose();
          end
          y = obj.doMult(x, dms);
          for k=1:length(obj.mtrx)
            obj.mtrx{k}.transpose();
          end
        end
        
        function mtrx = compMatrix(obj)
          mtrx = obj.mtrx{1}.getMatrix();
          for k=2:length(obj.mtrx)
            mtrx = kron(mtrx, obj.mtrx{k}.getMatrix());
          end
          
%           mtx_ref = obj.compMatrix@SensingMatrix();
%           if norm(mtrx(:)-mtx_ref(:),inf) >= 1e-10 * norm(mtrx(:))
%             error('getMatrix() returned a wrong result');
%           end
        end
          
    end
    methods (Access=protected)
        % Initialize the object
        %   Input:
        %     obj - this object
        %     mtrcs - the matrices sequence
        function setSensingMatrixKron(obj, mtrcs)
            switch nargin
                case 1
                    sm_args = {};
                case 2
                  % take only non-empty matrices
                  ne_mtrcs = cell(size(mtrcs));
                  cnt = 0;
                  for k=1:length(mtrcs)
                    if ~isempty(mtrcs{k})
                      cnt = cnt+1;
                      ne_mtrcs{cnt} = mtrcs{k};
                    end
                  end
                  sm_args = {ne_mtrcs(1:cnt)};
            end
            obj.setSensingMatrixComposed(sm_args{:});
        end

        function [ncl, nrw, tr_fctr, nnclp] = compDim(obj, mtrcs)
          obj.dims = zeros(length(mtrcs),2);
          tr_fctr = 1;
          for k=1:length(mtrcs)
            if mtrcs{k}.nNoClip
              error('matrices cannot have non-zero nNoClip');
            end
            kr = length(mtrcs)+1 - k;
            obj.dims(kr,1) = mtrcs{k}.nRows();
            obj.dims(kr,2) = mtrcs{k}.nCols();
            tr_fctr = tr_fctr * mtrcs{k}.trnspScale();
          end
          rwcl = prod(obj.dims);
          nrw = rwcl(1);
          ncl = rwcl(2);
          nnclp = 0;
        end
 
        % This function compute an upper bound, not the actual norm
        function nrm = compNormAtA(obj)
            n_mtrx = length(obj.mtrx);
            nrm_aa = zeros(n_mtrx,1);
            for k=1:n_mtrx
                nrm_aa(k) = obj.mtrx{k}.normAtA();
            end
            nrm = prod(nrm_aa);
        end
        
        function y = doMult(obj, x, dims)
          y = x;
          sz = dims(:,2)';
          for k=1:length(obj.mtrx);
            y = reshape(y(:),sz);
            prmt = 1:length(obj.mtrx);
            prmt([1,k]) = prmt([k,1]); % Swap indices 1,k
            y = permute(y,prmt);
            sz_prmt = size(y);
            y = reshape(y(:), [sz(k), numel(y)/sz(k)]);
            y = obj.mtrx{end+1-k}.multMat(y);
            sz_prmt(1) = dims(k,1);
            sz(k) = dims(k,1);
            y = reshape(y(:), sz_prmt);
            y = ipermute(y,prmt);
            y = y(:);
          end
        end
    end
end
