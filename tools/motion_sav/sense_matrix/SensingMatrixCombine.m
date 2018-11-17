classdef SensingMatrixCombine < SensingMatrixComposed
    %SensingMatrixCombine is a linear combination of several sensing
    %matrices of the same order
    %   Detailed explanation goes here
    
    properties
        wgt=[];   % a vector of n_mtrx weights
    end
    
    methods
        % Constructor can have 0,2 or 3 arguments.
        function obj = SensingMatrixCombine(varargin)
            obj.setSensingMatrixCombine(varargin{:})
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'wg', 'mtrcs', 'nrm_aa'});
            obj.setSensingMatrixCombine(varargin{:})
        end
        
        function setWgt(obj, wg)
            if iscell(wg)
                obj.wgt = zeros(size(wg));
                for k=1:numel(wg)
                    obj.wgt(k) = obj.wgt{k};
                end
            else
                obj.wgt = wg;
            end
        end
                        
        function len=encode(obj, code_dst, info)
            len = obj.encode@SensingMatrixComposed(code_dst, info);
            if ischar(len)
                return
            end
            len0 = code_dst.writeNumber([obj.wgt]);
            if ischar(len0)
                len = len0;
                return;
            end
            len = len + len0;
        end
        
        function len=decode(obj, code_src, info, cnt)
            len = obj.decode@SensingMatrixComposed(code_src, info, cnt);
            if ischar(len) || (isscalar(len) && len==-1)
                return;
            end
            
            n_mtrx = length(obj.mtrx);
            [wg, len0] = code_src.readNumber(cnt-len, [n_mtrx,1]);
            if ischar(len0)
                len = len0; return
            elseif isscalar(len0) && len0==-1
                len = 'EOD encountered'; return
            end
            len = len+len0;
            obj.setWgt(double(wg));
        end
        
        % doMultVec - Multiply a vector x of length nCols() by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = zeros(obj.n_rows, 1);
            for k=1:length(obj.wgt)
                y = y + obj.wgt(k)*obj.mtrx{k}.multVec(x);
            end
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length nCols().  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = zeros(obj.nCols(), 1);
            for k=1:length(obj.wgt)
                y = y + obj.wgt(k)*obj.mtrx{k}.multTrnspVec(x);
            end
        end
        
        function mtrx = compMatrix(obj)
          mtrx = obj.wgt(1) * obj.mtrx{1}.getMatrix();
          for k=2:length(obj.mtrx)
            mtrx = mtrx + obj.wgt(k)*obj.mtrx{k}.getMatrix();
          end
        end
    end
        
    methods (Access=protected)
        function setSensingMatrixCombine(obj, wg, mtrcs, nrm_aa)
            switch nargin
                case 1
                    sm_args = {};
                case 3
                    sm_args = {mtrcs};
                case 4
                    sm_args = {mtrcs, nrm_aa};
            end
            obj.setSensingMatrixComposed(sm_args{:});
             
            if nargin > 1
                obj.setWgt(wg);
            end
        end
        
        function [ncl, nrw, tr_fctr, nnclp] = compDim(~, mtrcs)
            nrw = mtrcs{1}.n_rows;
            ncl = mtrcs{1}.nCols();
            tr_fctr = mtrcs{1}.trnspScale();
            nnclp = mtrcs{1}.nNoClip();
            sorted = mtrcs{1}.sortNoClip((1:nrw)');
            for k=2:length(mtrcs)
                mtx = mtrcs{k};
                if nrw ~= mtx.n_rows || ncl ~= mtx.nCols()
                    error('not all matrices have same dimensions');
                end
                if tr_fctr ~= mtx.trnspScale()
                    error('not all matrices have same transpose factor');
                end
                
                if ~isequal(sorted, mtx.sortNoClip((1:nrw)'))
                    error('not all matrices have the same sortNoClip()');
                end
                mtx_nnclp = mtx.nNoClip();
                if mtx_nnclp > nnclp
                    nnclp = mtx_nnclp;
                end
            end
        end
 
        % This function compute an upper bound, not the actual norm
        function nrm = compNormAtA(obj)
            n_mtrx = length(obj.mtrx);
            nrm_aa = zeros(n_mtrx,1);
            for k=1:n_mtrx
                nrm_aa(k) = obj.normAtA{k}.normAtA()*(obj.wgt(k)^2);
            end
            nrm = sum(nrm_aa);
        end
    end
    
end

