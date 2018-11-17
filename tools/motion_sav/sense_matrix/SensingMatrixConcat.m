classdef SensingMatrixConcat < SensingMatrixComposed
    %SensingMatrixConcat Concatenation of several sensing matrices (the
    %measurement vector is the concatenation of the measurements vectors of
    %all matrices)
    %   
    
    properties
    end
    
    methods
        % Constructor can have 0,1 or 2 arguments.
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        %     nrm_aa (optional) norm A'A
        function obj = SensingMatrixConcat(varargin)
            obj.set(varargin{:})
        end
        
        %   Input
        %     obj - this object
        %     mtrcs - a cell array of matrices
        %     nrm_aa (optional) norm A'A
        function set(obj, varargin)
            if nargin == 2 && (isstruct(varargin{1}) || ischar(varargin{1}))
                spec = varargin{1};
                if ischar(varargin{1})
                    spec = ProcessingParams.parse_opts(spec);
                end
                if isfield(spec, 'mtrcs')
                    if isfield(spec, 'num_rows')
                        for k=1:length(spec.mtrcs)
                            if isstruct(spec.mtrcs{k}) && ...
                                    isfield(spec.mtrcs{k}.args, 'num_rows') &&...
                                    spec.mtrcs{k}.args.num_rows < 1
                                spec.mtrcs{k}.args.num_rows = ...
                                    floor(spec.num_rows * spec.mtrcs{k}.args.num_rows);
                            end
                        end
                    end
                    
                    if isfield(spec, 'num_columns')
                        spec.mtrcs = ...
                            SensingMatrixComposed.setMtrcsNCols(...
                            spec.mtrcs, spec.num_columns);
                    end
                    if isfield(spec, 'rnd_seed')
                        spec.mtrcs = ...
                            SensingMatrixComposed.setRndSeed(...
                            spec.mtrcs, spec.num_columns);
                    end
                    if isfield(spec, 'order')
                        spec.mtrcs = ...
                            SensingMatrixComposed.setMtrcsOrder(...
                            spec.mtrcs, spec.order);
                    end
                end
                varargin{1} = spec;
            end
            varargin = parseInitArgs(varargin, {'mtrcs', 'nrm_aa'});
            obj.setSensingMatrixConcat(varargin{:})
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = zeros(obj.n_rows, 1);
            bgn = 1;
            for k=1:length(obj.mtrx)
                yy = obj.mtrx{k}.multVec(x);
                new_bgn = bgn + length(yy);
                y(bgn:(new_bgn-1)) = yy;
                bgn = new_bgn;
            end
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = zeros(obj.n_cols, 1);
            bgn = 1;
            for k=1:length(obj.mtrx)
                new_bgn = bgn+obj.mtrx{k}.n_rows;
                y = y + obj.mtrx{k}.multTrnspVec(x(bgn:(new_bgn-1)));
                bgn = new_bgn;
            end
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
        
        % Get an array of measurements (or modified measurements)
        % which correspond to specific  offsets.
        %   Input
        %     obj: This object
        %     ofsts: a vector of offsets of length lofst.
        %     msrs: The measurements vector
        %     inp_list: Can be n array which cotains measurement indices of the
        %               measurements to use in msrs, or the string 'all'
        %     params: Optional struct of parameters which may be of use to some
        %             subclasses. Possible arguments include:
        %               nrm - norm with which comparison is to be done.
        %               ofsts_list - a column vector of indices of columns of
        %                           ofst_msrs (see below)
        %               nghbr_list - an array of indices of columns of
        %                            ofst_msrs. The number of rows is the
        %                            same as length(params.ofsts_list
        %  Output
        %     ofst_msrs: An array of size [m, lofts]. The i-th column
        %        contains the measurements (or modified measurements)
        %        corresponding to the offsets ofsts(i).
        %
        %  Note: If params is present and has the fields ofsts_list
        %        and nghbr_list, then after computing ofst_msrs it is
        %        modified by a call
        %    ofst_msrs = obj.getEdgeMsrmnts(ofst_msrs, 
        %                                    params.ofsts_list,
        %                                    params.nghbr_list);
        %        or something functionally equivalent. 
        function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts, msrs, ...
                inp_list, params)
            if obj.is_transposed
                ofst_msrs = zeros(0,length(ofsts));
                return
            end
            
            if nargin < 5
                params = struct();
            end
            
            if ~isnumeric(inp_list)
                inp_list = 1:obj.nRows();
            end
            
            % Eliminate measurements in the zeroed list
            [~,zr_ind,~] = intersect(inp_list,obj.zeroed_rows);
            inp_list(zr_ind)=[];
            
            msrs_end = 0;
            ofst_msrs_k = cell(1,length(obj.mtrx));
            for k=1:length(obj.mtrx)
                mtx = obj.mtrx{k};
                msrs_bgn = msrs_end+1;
                msrs_new_end = msrs_bgn + mtx.nRows() - 1;
                [inp_list_k, indcs_k] = intersect(inp_list, msrs_bgn:msrs_new_end);
                msrs_k = msrs(indcs_k);
                inp_list_k = inp_list_k - msrs_end;
                
                ofst_msrs_k{k} = ...
                    mtx.getOffsetMsrmnts(ofsts, msrs_k, inp_list_k, params);
                
                msrs_end = msrs_new_end;
            end
            ofst_msrs = vertcat(ofst_msrs_k{:});
        end
                
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          if obj.is_transposed
            y = obj.normalizeMsrs@SensingMatrix(y);
          else
            ybgn = 1;
            for k=1:length(obj.mtrx)
              mtx = obj.mtrx{k};
              yend = ybgn + mtx.nRows() - 1;
              y(ybgn:yend) = mtx.normalizeMsrs(y(ybgn:yend));
              ybgn = yend +1;
            end
          end
        end
                    
        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          if obj.is_transposed
            y = obj.deNormalizeMsrs@SensingMatrix(y);
          else
            ybgn = 1;
            for k=1:length(obj.mtrx)
              mtx = obj.mtrx{k};
              yend = ybgn + mtx.nRows() - 1;
              y(ybgn:yend) = mtx.deNormalizeMsrs(y(ybgn:yend));
              ybgn = yend +1;
            end
          end
        end
                    
        function mtrx = getMatrix(obj)
          mtrx = obj.mtrx{1}.getMatrix();
          for k=2:length(obj.mtrx)
            obj.mtrx = cat(1, mtrx, obj.mtrx{k}.getMatrix());
          end
        end
        
        % Get the DC measurement
        function dc_val = getDC(obj,msrs)
            dc_val = [];
            for k=1:length(obj.mtrx)
                dc_val = obj.mtrx{k}.getDC(msrs);
                if ~isempty(dc_val)
                    return
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
        function setSensingMatrixConcat(obj, mtrcs, nrm_aa)
            switch nargin
                case 1
                    sm_args = {};
                case 2
                    sm_args = {mtrcs};
                case 3
                    sm_args = {mtrcs, nrm_aa};
            end
            obj.setSensingMatrixComposed(sm_args{:});
        end

        function [ncl, nrw, tr_fctr, nnclp] = compDim(~, mtrcs)
            nrw = mtrcs{1}.n_rows;
            ncl = mtrcs{1}.n_cols;
            tr_fctr = mtrcs{1}.trnspScale();
            nnclp = mtrcs{1}.nNoClip();
            for k=2:length(mtrcs)
                if ncl ~= mtrcs{k}.n_cols
                    error('not all matrices have same number of columns');
                end
                if tr_fctr ~= mtrcs{k}.trnspScale()
                    error('not all matrices have same transpose factor');
                end
                nrw = nrw + mtrcs{k}.n_rows;
                nnclp = nnclp + mtrcs{k}.nNoClip();
            end
        end
 
    end

end

