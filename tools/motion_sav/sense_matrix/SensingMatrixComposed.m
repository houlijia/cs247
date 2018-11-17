classdef SensingMatrixComposed < SensingMatrix
    %SensingMatrixComposed An abstract class of sensing matrices composed
    %of other matrices.
    %   
    
    properties
        mtrx=[];  % Cell array n_mtrx of matrices
        trnsp_fctr = 1;
    end
    
    properties (Access=protected)
        n_no_clip = 0;
        nrm_ata = -1;
    end
    
    methods
        % Constructor can have 0,1 or 2 arguments.
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        %     nrm_aa (optional) norm A'A
        function obj = SensingMatrixComposed(varargin)
            varargin = parseInitArgs(varargin, {'mtrcs', 'nrm_aa'});
            obj.setSensingMatrixComposed(varargin{:})
        end
        
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        %     nrm_aa (optional) norm A'A
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'mtrcs','nrm_aa'});
            obj.setSensingMatrixComposed(varargin{:})
        end
        
        function eql = isEqual(obj, other)
            if class(obj) ~= class(other)
                eql = false;
                return
            end
            
            if length(obj.mtrx) ~= length(other.mtrx) ||...
                    obj.trnsp_fctr ~= other.trnsp_fctr ||...
                    obj.nrm_ata ~= other.nrm_ata ||...
                    obj.n_no_clip ~= other.n_no_clip
                eql = false;
                return
            end
            
            for k=1:length(obj.mtrx)
                if ~ obj.mtrx{k}.isEqual(other.mtrx{k})
                    eql = false;
                    return
                end
            end
            eql = true;
        end
        
        function init(obj, mtrcs)
            obj.set(mtrcs);
        end
        
        function setNrmAtA(obj, nrm_aa)
            obj.nrm_ata = nrm_aa;
        end
        
        function cnt = getContainedLength(obj)
            cnt = length(obj.mtrx);
            for k=1:length(obj.mtrx)
                cnt = cnt + obj.mtrx{k}.getContainedLength();
            end
        end
        
        function lst = getContained(obj)
            lst = cell(1, obj.getContainedLength());
            for k=1:length(obj.mtrx)
                lst{k} = obj.mtrx{k};
            end
            bgn = 1 + length(obj.mtrx);
            for k=1:length(obj.mtrx)
                sublst = obj.mtrx{k}.getContained();
                new_bgn = bgn + length(sublst);
                lst(bgn:new_bgn-1) = sublst;
                bgn = new_bgn;
            end
        end
                
        function len=encode(obj, code_dst, info)
            len = code_dst.writeUInt(length(obj.mtrx));
            if ischar(len)
                return
            end
            len0 = code_dst.writeNumber(obj.nrm_ata);
            if ischar(len0)
                len = len0;
                return;
            end
            len = len+len0;
            
            for k=1:length(obj.mtrx)
                len0 = obj.mtrx{k}.write(info, code_dst, true);
                if ischar(len0)
                    len = len0;
                    return;
                end
                len = len + len0;
            end
        end
        
        function len=decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            [n_mtrx, len] = code_src.readUInt(cnt);
            if ischar(len) || (isscalar(len) && len==-1)
                return;
            end
            
            [nrm_aa, len0] = code_src.readNumber(cnt-len);
            if ischar(len0)
                len = len0; return
            elseif isscalar(len0) && len0==-1
                len = 'EOD encountered'; return
            end
            len = len+len0;
            
            mtrcs = cell(1,n_mtrx);
            for k=1:n_mtrx
                [mtrcs{k}, len0,~] = CodeElement.readElement(info,code_src);
                if ischar(mtrcs{k})
                    len = mtrcs{k}; return;
                elseif isscalar(mtrcs{k}) && isnumeric(mtrcs{k}) && ...
                    mtrcs{k} == -1
                    len = 'EOD encountered'; return
                end
                len = len + len0;
                if len > cnt
                    len = 'Exceeded allowed number of bytes'; return
                end
            end
            
            obj.setMtrcs(mtrcs);
            obj.setNrmAtA(nrm_aa);
        end
        
        % Sometimes multTrnspVec may multiply the output y by a scaling
        % factor.  This function returns the scaling factor.
        function y = trnspScale(obj)
            y=obj.trnsp_fctr;
        end
        
        % returns the norm of A'A (transpose times the matrix the norm is
        % The norm is defined as max(obj.multTrnspVec(obj.multVec(x))) over 
        % all x such that x'x=1.
        function y = normAtA(obj)
            if obj.nrm_ata < 0
                obj.setNrmAtA(obj.compNormAtA());
            end
            y = obj.nrm_ata;
        end
        
        function n_no_clip = nNoClip(obj)
            if obj.is_transposed
                n_no_clip = 0;
            else
                n_no_clip = obj.n_no_clip;
            end
        end
    
        % returns true if getMatrix returns a sparse matrix
        function is_sprs = isSparse(obj)
            is_sprs = true;
            for k=1:length(obj.mtrx)
                if ~obj.mtrx{k}.isSparse()
                    is_sprs = false;
                    return;
                end
            end
        end
        
    end

    methods (Access=protected)
        % Initialize then object
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        %     nrm_aa (optional) norm A'A
        function setSensingMatrixComposed(obj, mtrcs, nrm_aa)
            obj.setSensingMatrix();
            
            if nargin > 1
                obj.setMtrcs(mtrcs);
                if nargin > 2
                    obj.setNrmAtA(nrm_aa);
                end
            end
        end
       
        function setMtrcs(obj, mtrcs)
            n = length(mtrcs);
            if ~n
                obj.mtrx = [];
                return;
            end
            
            for k=1:numel(mtrcs)
                if ~isa(mtrcs{k}, 'SensingMatrix')
                    mtrcs{k} = SensingMatrix.construct(...
                        mtrcs{k}.type, mtrcs{k}.args);
                end
            end
            [ncl, nrw, tr_fctr, nnclp] = obj.compDim(mtrcs);
            
            obj.setSensingMatrix(nrw, ncl);
            if ~isrow(mtrcs)
                mtrcs = mtrcs';
            end
            obj.mtrx = mtrcs;
            obj.trnsp_fctr = tr_fctr;
            obj.n_no_clip = nnclp;
        end
        
        function out = sortNoClipSect(obj, y)
            out = zeros(size(y));
            nc_bgn = 1;
            c_bgn = obj.n_no_clip+1;
            y_bgn = 1;
            
            for k=1:length(obj.mtrx)
                mtx = obj.mtrx{k};
                nr = mtx.n_rows;
                nc = mtx.nNoClip();
                
                nc_bgn_new = nc_bgn+nc;
                c_bgn_new = c_bgn + nr - nc;
                y_bgn_new = y_bgn + nr;
                
                ok = mtx.sortNoClip(y(y_bgn:y_bgn_new-1));
                out(nc_bgn:nc_bgn_new-1) = ok(1:nc);
                out(c_bgn:c_bgn_new-1) = ok(nc+1:nr);
                
                nc_bgn = nc_bgn_new;
                c_bgn = c_bgn_new;
                y_bgn = y_bgn_new;
            end
        end
            
        function out = unsortNoClipSect(obj, y)
            out = zeros(size(y));
            nc_bgn = 1;
            c_bgn = obj.n_no_clip+1;
            y_bgn = 1;
            
            for k=1:length(obj.mtrx)
                mtx = obj.mtrx{k};
                nr = mtx.n_rows;
                nc = mtx.nNoClip();
                
                nc_bgn_new = nc_bgn+nc;
                c_bgn_new = c_bgn + nr - nc;
                y_bgn_new = y_bgn + nr;
                
                ok = [y(nc_bgn:nc_bgn_new-1); y(c_bgn:c_bgn_new-1)];
                out(y_bgn:y_bgn_new-1) = mtx.unsortNoClip(ok);
                
                nc_bgn = nc_bgn_new;
                c_bgn = c_bgn_new;
                y_bgn = y_bgn_new;
            end
        end
        
        % Compute nrmAtA. This is exact for SensingMatrixBlkDiag and an
        % upper bound for SensingMatrixConcat
        function nrm = compNormAtA(obj)
            n_mtrx = length(obj.mtrx);
            nrm_aa = zeros(n_mtrx,1);
            for k=1:n_mtrx
                nrm_aa(k) = obj.mtrx{k}.normAtA();
            end
            nrm = max(nrm_aa);
        end
    end
    
    methods (Access=protected, Abstract)
        % Compute the matrix dimensions, 
        [ncl, nrw, tr_fctr, nnclp] = compDim(obj, mtrcs)
    end
    
    methods (Static, Access=protected)
        function mtrcs = setMtrcsNCols(mtrcs, n_cols)
            for k=1:length(mtrcs)
                if isstruct(mtrcs{k}) && ~isfield(mtrcs{k}.args, 'num_columns')
                    mtrcs{k}.args.num_columns = n_cols;
                end
            end
        end
        
        function mtrcs = setRndSeed(mtrcs, r_seed)
            for k=1:length(mtrcs)
                if isstruct(mtrcs{k}) && ~isfield(mtrcs{k}.args, 'r_seed')
                    mtrcs{k}.args.rnd_seed = r_seed+k;
                end
            end
        end
        
        function mtrcs = setMtrcsOrder(mtrcs, order)
            for k=1:length(mtrcs)
                if isstruct(mtrcs{k}) && ~isfield(mtrcs{k}.args, 'order')
                    mtrcs{k}.args.order = order;
                end
            end
        end
        
    end
end

