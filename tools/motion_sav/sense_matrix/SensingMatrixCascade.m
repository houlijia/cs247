classdef SensingMatrixCascade < SensingMatrixComposed
    %SensingMatrixCascade is a composition of several matrices.
    %   mtrx{1}*mtrx{2}*...
    
    properties
    end
    
    methods
        % Constructor
        %   Input:
        %     mtrcs - the matrices sequence
       function obj = SensingMatrixCascade(varargin)
            obj.setSensingMatrixCascade(varargin{:})
        end
        
        % Initialize the object
        %   Input:
        %     obj - this object
        %     mtrcs - the matrices sequence
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'mtrcs'});
            obj.setSensingMatrixCascade(varargin{:})
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = x;
            for k=length(obj.mtrx):-1:1
                y = obj.mtrx{k}.multVec(y);
            end
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = x;
            for k=1:length(obj.mtrx)
                y = obj.mtrx{k}.multTrnspVec(y);
            end
        end
        
        % Sort the measurements vector y so that the no clip elements are first.
        % If the function is called without argument it returns a handle to
        % itself.
        function out = sortNoClip(obj, y)
            if nargin < 2
                out = @(yy) obj.mtrx{1}.sortNoClip(yy);
            elseif obj.is_transposed
                out = y;    
            else
                out = obj.mtrx{1}.sortNoClip(y);
            end
        end
        
        % Unsort the sorted vector y so that the no clip elements are in 
        % their original place.
        % If the function is called without argument it returns a handle to
        % itself.
        function out = unsortNoClip(obj, y)
            if nargin < 2
                out = @(yy) obj.mtrx{1}.unsortNoClip(yy);
            elseif obj.is_transposed
                out = y;    
            else
                out = obj.mtrx{1}.unsortNoClip(y);
            end
        end
        
        function mtrx = compMatrix(obj)
          mtrx = obj.mtrx{1}.getMatrix();
          for k=2:length(obj.mtrx)
            mtrx = mtrx * obj.mtrx{k}.getMatrix();
          end
        end
    
        % Get the DC measurement (taking it based on first
        % matrix in the sequence).
        function dc_val = getDC(obj,msrs)
            dc_val = obj.mtrx{1}.getDC(msrs);
        end
    end

    methods (Access=protected)
        % Initialize the object
        %   Input:
        %     obj - this object
        %     mtrcs - the matrices sequence
        function setSensingMatrixCascade(obj, mtrcs)
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

        function [ncl, nrw, tr_fctr, nnclp] = compDim(~, mtrcs)
            nrw = mtrcs{1}.nRows();
            ncl = mtrcs{end}.nCols();
            tr_fctr = mtrcs{1}.trnspScale();
            nnclp = mtrcs{1}.nNoClip();
            for k=2:length(mtrcs)
                if mtrcs{k}.nNoClip
                    error('only first matrix can have non-zero nNoClip');
                end
                if mtrcs{k}.nRows() ~= mtrcs{k-1}.nCols()
                    error('mtrcs{%d}.nRows()=%d ~= mtrcs{%d}.nCols()=%d',...
                        k, mtrcs{k}.nRows(), k-1, mtrcs{k-1}.nCols());
                end
                tr_fctr = tr_fctr * mtrcs{k}.trnspScale();
            end
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
    end


end

