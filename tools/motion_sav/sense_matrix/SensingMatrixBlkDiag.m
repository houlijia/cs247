classdef SensingMatrixBlkDiag < SensingMatrixComposed
    %SensingMatrixBlkDiag Concatenation of several sensing matrices (the
    %measurement vector is the concatenation of the measurements vectors of
    %all matrices)
    %   
    
    properties
    end
    
    methods
        % Constructor can have 0 or 1 arguments.
        %   Input
        %     mtrcs - a cell array of matrices
        function obj = SensingMatrixBlkDiag(varargin)
            obj.setSensingMatrixBlkDiag(varargin{:})
        end
        
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'mtrcs'});
            obj.setSensingMatrixBlkDiag(varargin{:})
        end
        
        % multVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = zeros(obj.n_rows, 1);
            xbgn = 1;
            ybgn = 1;
            for k=1:length(obj.mtrx)
                new_xbgn = xbgn + obj.mtrx{k}.nCols();
                new_ybgn = ybgn + obj.mtrx{k}.nRows();
                y(ybgn:(new_ybgn-1)) = ...
                    obj.mtrx{k}.multVec(x(xbgn:(new_xbgn-1)));
                xbgn = new_xbgn;
                ybgn = new_ybgn;
            end
         end
        
        % multTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = zeros(obj.n_cols, 1);
            xbgn = 1;
            ybgn = 1;
            for k=1:length(obj.mtrx)
                new_xbgn = xbgn + obj.mtrx{k}.nRows();
                new_ybgn = ybgn + obj.mtrx{k}.nCols();
                y(ybgn:(new_ybgn-1)) = ...
                    obj.mtrx{k}.multTrnspVec(x(xbgn:(new_xbgn-1)));
                xbgn = new_xbgn;
                ybgn = new_ybgn;
            end
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          ybgn = 1;
          for k=1:length(obj.mtrx)
            mtx = obj.mtrx{k};
            if obj.is_transposed
              mtx = mtx.transpose();
            end
            yend = ybgn + mtx.nRows() - 1;
            y(ybgn:yend) = mtx.normalizeMsrs(y(ybgn:yend));
            ybgn = yend +1;
          end
        end
        
        % undo the operation of normalizeMsrs
        function y=deNormalizeMsrs(obj,y)
          ybgn = 1;
          for k=1:length(obj.mtrx)
            mtx = obj.mtrx{k};
            if obj.is_transposed
              mtx = mtx.transpose();
            end
            yend = ybgn + mtx.nRows() - 1;
            y(ybgn:yend) = mtx.deNormalizeMsrs(y(ybgn:yend));
            ybgn = yend +1;
          end
        end
            
        % Sometimes multTrnspVec may multiply the output y by a scaling
        % factor.  This function returns the scaling factor.
        function y = trnspScale(obj)
            y=obj.trnsp_fctr;
        end
        
        % Sort the measurements vector y so that the no clip elements are first.
        % If the function is called without argument it returns a handle to
        % itself.
        function out = sortNoClip(obj, y)
            if nargin < 2
                out = @(yy) obj.sortNoClip(yy);
            elseif obj.is_transposed
                out = y;    
            else
                out = obj.sortNoClipSect(y);
            end
        end
        
        % Unsort the sorted vector y so that the no clip elements are in 
        % their original place.
        % If the function is called without argument it returns a handle to
        % itself.
        function out = unsortNoClip(obj, y)
            if nargin < 2
                out = @(yy) obj.unsortNoClip(yy);
            elseif obj.is_transposed
                out = y;    
            else
                out = obj.unsortNoClipSect(y);
            end
        end
        
        % Redefine getMtrx as getting a sparse matrix
        function mtrx = compMatrix(obj)
          if obj.isSparse()
            r_bgn = 0;
            c_bgn = 0;
            n_mtrx = length(obj.mtrx);
            rr = cell(n_mtrx,1);
            cc = cell(n_mtrx,1);
            vv = cell(n_mtrx,1);
            n_ttl = 0;
            for k=1:n_mtrx
              [r,c,v] = find(obj.mtrx{k});
              rr{k} = r + r_bgn;
              cc{k} = c + c_bgn;
              vv{k} = v;
              r_bgn = r_bgn + obj.mtrx{k}.nRows();
              c_bgn = c_bgn + obj.mtrx{k}.nCols();
              n_ttl = n_ttl + length(r);
            end
            r = zeros(n_ttl,1);
            c = zeros(n_ttl,1);
            v = zeros(n_ttl,1);
            i0 = 1;
            for k=1:n_mtrx
              i1 = i0 + length(rr{k})-1;
              r(i0:i1) = rr{k};
              c(i0:i1) = cc{k};
              v(i0:i1) = vv{k};
              i0 = i1+1;
            end
            mtrx = sparse(r,c,v,obj.n_rows,obj.n_cols);
          else
            mtrx = obj.compMatrix@SensingMatrix();
          end
        end
        
        % Get the DC measurement
        function dc_val = getDC(obj,msrs)
            dc_val = 0;
            xbgn = 1;
            for k=1:length(obj.mtrx)
                mtx = obj.mtrx{k};
                if ~mtx.nCols()
                    continue;
                end
                new_xbgn = xbgn + mtx.nCols();
                dcv = mtx.getDC(msrs(xbgn:new_xbgn-1));
                if isempty(dcv)
                    dc_val = [];
                    return;
                end
                dc_val = dc_val + dcv;
                xbgn = new_xbgn;
            end
        end
        
    end

    methods (Access=protected)
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        function setSensingMatrixBlkDiag(obj, mtrcs)
            if nargin == 1
                sm_args = {};
            else
                sm_args = {mtrcs};
            end
            obj.setSensingMatrixComposed(sm_args{:});
        end
        
        function [ncl, nrw, tr_fctr, nnclp] = compDim(~, mtrcs)
            nrw = mtrcs{1}.nRows();
            ncl = mtrcs{1}.nCols();
            tr_fctr = mtrcs{1}.trnspScale();
            nnclp = mtrcs{1}.nNoClip();
            for k=2:length(mtrcs)
                if tr_fctr ~= mtrcs{k}.trnspScale()
                    error('not all matrices have same transpose factor');
                end
                ncl = ncl + mtrcs{k}.nCols();
                nrw = nrw + mtrcs{k}.nRows();
                nnclp = nnclp + mtrcs{k}.nNoClip();
            end
        end
    end
end

