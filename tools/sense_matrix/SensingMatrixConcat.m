classdef SensingMatrixConcat < SensingMatrixComposed
    %SensingMatrixConcat Concatenation of several sensing matrices (the
    %measurement vector is the concatenation of the measurements vectors of
    %all matrices)
    %   
    
    properties (Access = private)
      % Begin and end indices of the vector components corresponding to each
      % component matrix. 
      mtrx_bgn;
      mtrx_end
    end
  
  
  methods
        function obj = SensingMatrixConcat(varargin)
          % Constructor can have 0,1 or 2 arguments.
          %   Input
          %     mtrcs: a cell array of matrices. Each cell contains either a matrix
          %       object or struct specifying a matrix (a struct with fields
          %       type' and 'args').  Alternatively, if there is only one arguments
          %       and this is a struct, then htis argument is the opts argument, with an
          %       additional mtrcs field.
          %     opts: A struct which may contain the following optional
          %     fields:
          %       nrm_aa: A norm to be set by setNorm()
          %       n_rows: Specifies the number of rows. Normally it should equal
          %         the sum of rows in all component matrices. However, if some
          %         components matrices are specified by a struct and the number
          %         of rows is missing, then n_rows minus the sum of the number
          %         of rows in the component matrices in which it is defined is
          %         divided more-or-less equally among the matrices in which it 
          %         is not defined.
          %      min_row_ratio: If present, an array of the same size as mtrcs,
          %         but all entries are ignored except for those in which the
          %         number of rows is not defined. When allocating rows to these
          %         component matrices, the k-th matrix gets at least
          %         min_row_ratio(k)*n_cols. If n_rows is defined and there is
          %         not enough rows to allocate to everyone, the allocation is
          %         decreased more or less equally
          %      row_share: If present, an array of non-negative integers of the
          %          same size as mtrcs,
          %         but all entries are ignored except for those in which the
          %         number of rows is not defined. After allocating rows
          %         according to min_row_ratio, the remaining rows are allocated
          %         in proportion to row_share.
          %     prmt: A struct with fields such as PL_range or N_msrs
          %     normalize: controls normalization of measurements of each component
          %       matrix: 
          %         0 - no normalizaiton (default)
          %         1 - divide measurements by norm()
          %         2 - divide measurements by getExactNorm()
            obj.set(varargin{:})
        end
        
        function set(obj, varargin)
          %   Input
          %     obj - this object
          %     mtrcs: a cell array of matrices. Each cell contains either a matrix
          %       object or struct specifying a matrix (a struct with fields
          %       type' and 'args').  Alternatively, if there are only two arguments
          %       and this is a struct, then htis argument is the opts argument, with an
          %       additional mtrcs field.
          %     opts: A struct which may contain the following optional
          %     fields:
          %       nrm_aa: A norm to be set by setNorm()
          %       n_rows: Specifies the number of rows. Normally it should equal
          %         the sum of rows in all component matrices. However, if some
          %         components matrices are specified by a struct and the number
          %         of rows is missing, then n_rows minus the sum of the number
          %         of rows in the component matrices in which it is defined is
          %         divided more-or-less equally among the matrices in which it 
          %         is not defined.
          %      min_row_ratio: If present, an array of the same size as mtrcs,
          %         but all entries are ignored except for those in which the
          %         number of rows is not defined. When allocating rows to these
          %         component matrices, the k-th matrix gets at least
          %         min_row_ratio(k)*n_cols. If n_rows is defined and there is
          %         not enough rows to allocate to everyone, the allocation is
          %         decreased more or less equally
          %      row_share: If present, an array of non-negative integers of the
          %          same size as mtrcs,
          %         but all entries are ignored except for those in which the
          %         number of rows is not defined. After allocating rows
          %         according to min_row_ratio, the remaining rows are allocated
          %         in proportion to row_share.
          %     prmt: A struct with fields such as PL_range or N_msrs
          %     normalize: controls normalization of measurements of each component
          %       matrix: 
          %         0 - no normalizaiton (default)
          %         1 - divide measurements by norm()
          %         2 - divide measurements by getExactNorm()
          if nargin < 2
            return
          end
          
          obj.setSensingMatrixConcat(varargin{:});
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
%           y = obj.zeros(obj.n_rows, 1);
%           bgn = 1;
%           for k=1:length(obj.mtrx)
%             yy = obj.mtrx{k}.multVec(x);
%             new_bgn = bgn + length(yy);
%             y(bgn:(new_bgn-1)) = yy;
%             bgn = new_bgn;
%           end
%           y_ref = y;

          y = arrayfun(@(mtx) {mtx{1}.multVec(x)}, obj.mtrx(:));
          y = vertcat(y{:});
%           if ~isequal(y, y_ref)
%             error('Multiplication error');
%           end
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
%           y = obj.zeros(obj.n_cols, 1);
%           bgn = 1;
%           for k=1:length(obj.mtrx)
%             new_bgn = bgn+obj.mtrx{k}.nRows();
%             y = y + obj.mtrx{k}.multTrnspVec(x(bgn:(new_bgn-1)));
%             bgn = new_bgn;
%           end
%           y_ref = y;

          if obj.use_gpu
            y = arrayfun(@(k) ...
              {obj.mtrx{k}.multTrnspVec(x(obj.mtrx_bgn(k):obj.mtrx_end(k)))},...
              (1:numel(obj.mtrx)));
            
          else
            y = arrayfun(@(mtx,xbgn,xend) {mtx{1}.multTrnspVec(x(xbgn:xend))}, ...
              obj.mtrx(:), obj.mtrx_bgn(:), obj.mtrx_end(:));
          end
          y = sum(horzcat(y{:}),2);

%           if ~isequal(y, y_ref)
%             error('Multiplication error');
%           end
        end
        
        % doMultMat - Multiply a matrix x with n_cols rows by the matrix and return
        % a vector y of length n_rows.
        function y = doMultMat(obj, x)
%           y = obj.zeros(obj.n_rows, size(x,2));
%           bgn = 1;
%           for k=1:length(obj.mtrx)
%             yy = obj.mtrx{k}.multMat(x);
%             new_bgn = bgn + size(yy,1);
%             y(bgn:(new_bgn-1),:) = yy;
%             bgn = new_bgn;
%           end
%           y_ref = y;
          
          y = arrayfun(@(mtx) {mtx{1}.multMat(x)}, obj.mtrx(:));
          y = vertcat(y{:});
%           if ~isequal(y, y_ref)
%             error('Multiplication error');
%           end
        end
        
        % doMultTrnspMat - Multiply a matrix x with n_rows rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspMat(obj, x)
%           y = obj.zeros(obj.n_cols, size(x,2));
%           bgn = 1;
%           for k=1:length(obj.mtrx)
%             new_bgn = bgn+obj.mtrx{k}.nRows();
%             y = y + obj.mtrx{k}.multTrnspMat(x(bgn:(new_bgn-1),:));
%             bgn = new_bgn;
%           end
%           y_ref = y;

          if obj.use_gpu
            y = arrayfun(@(k) ...
              {obj.mtrx{k}.multTrnspVec(x(obj.mtrx_bgn(k):obj.mtrx_end(k),:))},...
              (1:numel(obj.mtrx)));
            
          else
            y = arrayfun(@(mtx,xbgn,xend) {mtx{1}.multTrnspMat(x(xbgn:xend,:))}, ...
              obj.mtrx(:), obj.mtrx_bgn(:), obj.mtrx_end(:));
          end
          y = sum(cat(3,y{:}),3);

%           if ~isequal(y, y_ref)
%             error('Multiplication error');
%           end
        end
        
        % Multiply a matrix or a vector by the matrix whose entries are the
        % absolute values of the entries of this matrix.
        function y = multAbs(obj, x)
          y = obj.zeros(obj.nRows(), size(x,2));
          bgn = 1;
          if ~obj.is_transposed
            for k=1:length(obj.mtrx)
              new_bgn = bgn + obj.mtrx{k}.nRows();
              y(bgn:(new_bgn-1),:) = ...
                obj.mtrx{k}.multAbs(x);
              bgn = new_bgn;
            end
          else
            for k=1:length(obj.mtrx)
              obj.mtrx{k}.transpose();
              new_bgn = bgn + obj.mtrx{k}.nCols();
              y = y + obj.mtrx{k}.multAbs(x(bgn:(new_bgn-1),:));
              bgn = new_bgn;
              obj.mtrx{k}.transpose();
            end
          end
        end
        
        % Computes the matlab matrix which corresponds to what this matrix
        % should be.
        function mtrx = doCompMatrix(obj)
          r_bgn = 0;
          n_mtrx = length(obj.mtrx);
          if obj.isSparse() && ~obj.use_gpu && ~obj.use_single
            rr = cell(n_mtrx,1); % row indices
            cc = cell(n_mtrx,1); % column indices
            vv = cell(n_mtrx,1); % values
            n_ttl = 0;
            for k=1:n_mtrx
              [r,c,v] = find(obj.mtrx{k}.getMatrix());
              if isa(v,'gpuArray')
                v = gather(v);
              end
              rr{k} = r + r_bgn;
              cc{k} = c;
              vv{k} = v;
              r_bgn = r_bgn + obj.mtrx{k}.nRows();
              n_ttl = n_ttl + length(r);
            end
            r = double(vertcat(rr{:}));
            c = double(vertcat(cc{:}));
            v = double(vertcat(vv{:}));
            if ~obj.is_transposed
              mtrx = sparse(r,c,v,double(obj.n_rows), double(obj.n_cols));
            else
              mtrx = sparse(c,r,v,double(obj.n_cols), double(obj.n_rows));
            end
          else
            mtrx = zeros(obj.nRows(), obj.nCols());
            for k=1:n_mtrx
              mtx = obj.mtrx{k}.getMatrix();
              if isa(mtx, 'gpuArray')
                mtx = gather(mtx);
              end
              r_end = r_bgn + size(mtx,1);
              if obj.is_transposed
                mtrx(:, r_bgn+1:r_end) = mtx';
              else
                mtrx(r_bgn+1:r_end, :) = mtx;
              end
              r_bgn = r_end;
            end
          end
        end
        
        
        function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts, msrs, ...
            inp_list, params)
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
            inp_list_k = intersect(inp_list, msrs_bgn:msrs_new_end);
            inp_list_k = inp_list_k - msrs_end;
            msrs_end = msrs_new_end;
            msrs_k = msrs(msrs_bgn:msrs_end);
            ofst_msrs_k{k} = ...
              mtx.getOffsetMsrmnts(ofsts, msrs_k, inp_list_k, params);
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
        
        % Returns the sum of values of the measurement which contain DC value,
        % weighted by the ratio of the DC value to other components (in
        % terms of RMS), or 0 if there is no such measurement.
        %   Input:
        %     obj - this object
        %     msrs - the measurements vector
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
    function setSensingMatrixConcat(obj, mtrcs, opts)
      % Initialize then object
      %   Input
      %     obj: This object
      %     mtrcs: a cell array of matrices. Each cell contains either a matrix
      %       object or struct specifying a matrix (a struct with fields
      %       type' and 'args'). Alternatively, if there are only two arguments
      %       and this is a struct, then htis argument is the opts argument, with an
      %       additional mtrcs field.
      %     opts: A struct which may contain the following optional
      %     fields:
      %       nrm_aa: A norm to be set by setNorm()
      %       n_rows: Specifies the number of rows. Normally it should equal
      %         the sum of rows in all component matrices. However, if some
      %         components matrices are specified by a struct and the number
      %         of rows is missing, then n_rows minus the sum of the number
      %         of rows in the component matrices in which it is defined is
      %         divided more-or-less equally among the matrices in which it
      %         is not defined.
      %      min_row_ratio: If present, an array of the same size as mtrcs,
      %         but all entries are ignored except for those in which the
      %         number of rows is not defined. When allocating rows to these
      %         component matrices, the k-th matrix gets at least
      %         min_row_ratio(k)*n_cols. If n_rows is defined and there is
      %         not enough rows to allocate to everyone, the allocation is
      %         decreased more or less equally
      %      row_share: If present, an array of non-negative integers of the
      %          same size as mtrcs,
      %         but all entries are ignored except for those in which the
      %         number of rows is not defined. After allocating rows
      %         according to min_row_ratio, the remaining rows are allocated
      %         in proportion to row_share.
      %     prmt: A struct with fields such as PL_range or N_msrs
      %     normalize: controls normalization of measurements of each component
      %       matrix:
      %         0 - no normalizaiton (default)
      %         1 - divide measurements by norm()
      %         2 - divide measurements by getExactNorm()
      
      if nargin <= 1
        return
      elseif nargin == 2
        if isstruct(mtrcs)
          opts = rmfield(mtrcs, 'mtrcs');
          mtrcs = mtrcs.mtrcs;
        else
          opts = struct();
        end
      end
      
      if ~isfield(opts, 'min_row_ratio')
        opts.min_row_ratio = zeros(size(mtrcs));
      end
      if isfield(opts, 'n_rows')
        if ~isfield(opts, 'row_share')
          opts.row_share = ones(size(mtrcs));
        end
      elseif isfield(opts, 'row_share')
        error('cannot specify row_share without n_rows in opts');
      end
      if ~isfield(opts, 'prmt')
        opts.prmt = struct();
      end
      if ~isfield(opts, 'normalize')
        opts.normalize = 0;
      end
      
      n_mtrcs = length(mtrcs);
      
      % Make sure that mtrcs, if structs, has args field
      for k=1:n_mtrcs
        if isstruct(mtrcs{k}) && ~isfield(mtrcs{k}, 'args')
          mtrcs{k}.args = struct();
        end
      end
      
      % find dims
      if ~isfield(opts, 'dims')
        for k=1:n_mtrcs
          mtx = mtrcs{k};
          if isstruct(mtx) && isfield(mtx.args, 'dims')
            opts.dims = mtx.args.dims;
          elseif isa(mtx, 'SensingMatrixMD')
            opts.dims = obj.dims;
          else
            continue;
          end
          break;
        end
      end
      
      if isfield(opts, 'dims')
        nc = prod(opts.dims);
        if ~isfield(opts,'n_cols')
          opts.n_cols = nc;
        elseif opts.n_cols ~= nc
          error('n_cols=%d ~= prod(dims)=prod(%s)=%d', opts.n_cols, ...
            show_str(opts.dimsa),nc);
        end
        
        for k=1:n_mtrcs
          if isstruct(mtrcs{k})
            if ~isfield(mtrcs{k}.args, 'dims')
              mtrcs{k}.args.dims = opts.dims;
            elseif ~isequal(double(opts.dims(:)), double(mtrcs{k}.args.dims(:)))
              error('opts.dims=%s while mtrcs{%d}.args.dims=%s',...
                show_str(opts.dims), k, show_str(mtrcs{k}.args.dims(:)));
            end
          elseif isa(mtrcs{k}, 'SensingMatrixMD') && ...
              ~isequal(double(opts.dims(:)), double(mtrcs{k}.dims(:)))
            error('opts.dims=%s while mtrcs{%d}.args.dims=%s',...
              show_str(opts.dims), k, show_str(mtrcs{k}.dims(:)));
          end
        end
      end
      
      % find n_cols
      if ~isfield(opts, 'n_cols')
        for k=1:n_mtrcs
          if isa(mtrcs{k}, 'SensingMatrix')
            opts.n_cols = mtrcs{k}.nCols();
            break;
          else
            if isfield(mtrcs{k}.args, 'transpose') && ...
                mtrcs{k}.args.transpose && isfield(mtrcs{k}.args, 'n_rows')
              opts.n_cols = mtrcs{k}.args.n_rows;
              break;
            elseif  isfield(mtrcs{k}.args, ' n_cols')
              opts.n_cols = mtrcs{k}.args.n_cols;
              break;
            end
          end
        end
        if ~isfield(opts, 'n_cols')
          error('no definitions for n_cols');
        end
      end
      nr_alloc = 0;
      nm_min_ratio = 0;
      nr_min_ratio = 0;
      ind_min_ratio = zeros(size(mtrcs));
      nr_share_ttl = 0;
      ind_share_ttl = zeros(size(mtrcs));
      
      for k=1:n_mtrcs
        if isstruct(mtrcs{k})
          % See if n_rows can be set using N_msrs
          if isfield(mtrcs{k}.args,'prmt') && ...
              isfield(mtrcs{k}.args.prmt, 'N_msrs')
            if ~isfield(mtrcs{k}.args, 'n_rows')
              mtrcs{k}.args.n_rows = mtrcs{k}.args.prmt.N_msrs;
            elseif mtrcs{k}.args.n_rows ~= mtrcs{k}.args.prmt.N_msrs
              error('mtrc{%d}.args: n_rows=d, prmt.N_msrs = %d',...
                k, mtrcs{k}.args.n_rows, mtrcs{k}.args.prmt.N_msrs)
            end
          end
          
          % try to set n_cols
          if isfield(mtrcs{k}.args, 'transpose') && mtrcs{k}.args.transpose
            if ~isfield(mtrcs{k}.args, 'n_rows')
              mtrcs{k}.args.n_rows = opts.n_cols;
            end
          else
            if ~isfield(mtrcs{k}.args, 'n_cols')
              mtrcs{k}.args.n_cols = opts.n_cols;
            end
          end
          
          % Check if the matrix can be constructed.
          try
            mtx = SensingMatrix.construct(mtrcs{k}.type, mtrcs{k}.args);
          catch excpt
            if isfield(mtrcs{k}.args, 'n_rows')
              rethrow(excpt);
            end
            mtx = [];
          end
          if ~isempty(mtx) && mtx.nRows()>0 && mtx.nCols()>0
            mtrcs{k} = mtx;
          end
        end
        
        if isa(mtrcs{k}, 'SensingMatrix')
          nr_alloc = nr_alloc + mtrcs{k}.nRows();
          continue;
        end
        
        if opts.min_row_ratio(k) > 0
          nm_min_ratio = nm_min_ratio + 1;
          ind_min_ratio(k) = floor(opts.n_cols * opts.min_row_ratio(k));
          nr_min_ratio = nr_min_ratio + ind_min_ratio(k);
        end
        
        if isfield(opts, 'row_share')
          if opts.row_share(k) > 0
            ind_share_ttl(k) = opts.row_share(k);
            nr_share_ttl = nr_share_ttl + ind_share_ttl(k);
          end
        end
      end
      
      if isfield(opts, 'n_rows')
        if nr_alloc > opts.n_rows
          error('total nRows() of mtrcs = %d > n_rows = %d', nr_alloc, opts.n_rows);
        elseif nr_alloc + nr_min_ratio > opts.n_rows
          excs = nr_alloc + nr_min_ratio - opts.n_rows;
          nr_min_ratio = nr_min_ratio - excs;
          d_excs = floor(excs / nm_min_ratio);
          r_excs = mod(excs, nm_min_ratio);
          for k=1:n_mtrcs
            if opts.min_row_ratio(k) == 0
              continue; 
            end
            rmdr = min(1,max(0,r_excs));
            ind_min_ratio(k) = ind_min_ratio(k) - d_excs - rmdr;
            r_excs = r_excs - rmdr;
          end
        end
      end
      
      for k=1:n_mtrcs
        if opts.min_row_ratio(k) == 0
          continue;
        end
        mtrcs{k}.args.n_rows = ind_min_ratio(k);
      end
        
      if isfield(opts, 'n_rows')
        avail = opts.n_rows - nr_alloc - nr_min_ratio;
        d_avail = floor(avail / nr_share_ttl);
        r_avail = mod(avail, nr_share_ttl);
        for k=1:n_mtrcs
          if opts.row_share(k) == 0
            continue;
          end
          rmdr = min(ind_share_ttl(k),max(0,r_avail));
          nr = d_avail * ind_share_ttl(k) + rmdr;
          r_avail = r_avail - rmdr;
          if isfield(mtrcs{k}.args, 'n_rows')
            mtrcs{k}.args.n_rows = mtrcs{k}.args.n_rows + nr;
          else
            mtrcs{k}.args.n_rows = nr;
          end
        end
      end

      % Ignore empty matrices
      ignore = false(size(mtrcs));
      for k=1:n_mtrcs
        if isstruct(mtrcs{k})
          ignore(k) = ...
            (~isfield(mtrcs{k}.args, 'n_rows') || mtrcs{k}.args.n_rows == 0);
        else
          ignore(k) = (mtrcs{k}.nRows() == 0);
        end
      end
      mtrcs(ignore) = [];
      n_mtrcs = length(mtrcs);
            
      % Create matrices
      for k=1:n_mtrcs
        if isstruct(mtrcs{k})
          mtx = SensingMatrix.construct(mtrcs{k}.type, mtrcs{k}.args);
        else
          mtx = mtrcs{k};
        end
        if mtx.nCols() ~= opts.n_cols
          error('mtrcs{%d}.nCols()=%d ~= opts.nCols=%d', k, mtx.nCols(), ...
            opts.n_cols);
        end
        if opts.normalize
          if opts.normalize == 1
            mtx_nrm = mtx.norm();
          else
            mtx_nrm = mtx.getExactNorm();
          end
          
          if mtx_nrm ~= 0 && mtx_nrm ~= 1
            mtx = SensingMatrixCascade.construct(...
              {SensingMatrixScaler(mtx.nRows(),(1/mtx_nrm)), mtx});
          end
          mtrcs{k} = mtx;
        end
      end
      
      if isfield(opts, 'nrm_aa')
        sm_args = {mtrcs, opts.nrm_aa};
      else
        sm_args = {mtrcs};
      end
       obj.setSensingMatrixComposed(sm_args{:});
      
      mtrx_len = ...
        arrayfun(@(k) obj.mtrx{k}.nRows(), (1:numel(obj.mtrx))');
      
      obj.mtrx_end = obj.toIndex(cumsum(mtrx_len));
      obj.mtrx_bgn = obj.toIndex(1 + [0; obj.mtrx_end(1:end-1,:)]);
      
      function indcell = get_indcs(k, trp)
        indcell = obj.mtrx{k}.indcsNoClip(trp)';
        if ~trp
          indcell = indcell + obj.mtrx_bgn(k) - 1;
        end
        indcell = {indcell};
      end
      
      % Compute no_clip info
      indcs = arrayfun(@(k) get_indcs(k, false), (1:numel(obj.mtrx))');
      indcs = vertcat(indcs{:});
      obj.setIndcsNoClip(indcs, false);
      
      indcs = arrayfun(@(k) get_indcs(k, true), (1:numel(obj.mtrx))');
      indcs = vertcat(indcs{:});
      obj.setIndcsNoClip(unique(indcs), true);
      
    end
    
    function [ncl, nrw, orth, psd_orth] = compDim(~, mtrcs)
      nrw = mtrcs{1}.nRows();
      ncl = mtrcs{1}.nCols();
      n_mtrcs = length(mtrcs);
      orth = struct('col', mtrcs{1}.getOrthoCol(), 'row',  ...
        (n_mtrcs==1 && mtrcs{1}.getOrthoRow()));
      psd_orth = struct('col', mtrcs{1}.getPsdOrthoCol(), 'row',  ...
        (n_mtrcs==1 && mtrcs{1}.getPsdOrthoRow()));
      for k=2:n_mtrcs
        if ncl ~= mtrcs{k}.nCols()
          error('not all matrices have same number of columns');
        end
        nrw = nrw + mtrcs{k}.nRows();
        orth.col = orth.col && mtrcs{k}.getOrthoCol();
        psd_orth.col = psd_orth.col && mtrcs{k}.getPsdOrthoCol();
      end
    end
    
    % Compute norm. This is an upper bound
    function nrm = compNorm(obj)
      n_mtrx = length(obj.mtrx);
      nrm_aa = arrayfun(@(k) obj.mtrx{k}.norm(), (1:n_mtrx)');
      nrm = norm(nrm_aa);
    end
    
    function mtx = create(obj, args)
      mtx = obj.construct(args);
    end
  end
  
  
  methods(Static)
    function mtx = construct(mtrcs, opts)
      if nargin < 2
        if ~isstruct(mtrcs)
          % mtrcs is a cell array. Set opts to empty struct
          opts = struct();
        else
          % mtrcs is an options struct, which contains all the options plus
          % a 'mtrcs' field, which contains the matrices cell array
          opts = mtrcs;
          mtrcs = opts.mtrcs;
          opts = rmfield(opts, 'mtrcs');
        end
      end
      
      % Check if any of the cells in matrices contains a struct, which
      % specifies the matrix (rather than the matrix itself). If so, check
      % if setting more args is necessary
      for k=1:numel(mtrcs)
        if ~isstruct(mtrcs{k})
          continue;
        end
        if ~isfield(mtrcs{k}, 'args')
          mtrcs{k}.args = struct();
        end
        if isfield(opts,'n_cols')
          mtrcs{k}.args.n_cols = opts.n_cols;
        end
        if isfield(opts, 'dims')
          mtrcs{k}.args.dims = opts.dims;
        end
        if isfield(opts, 'rnd_seed') && ~isfield(mtrcs{k}, 'rnd_seed')
          mtrcs{k}.args.rnd_seed = opts.rnd_seed + k;
        end
        if isfield(opts, 'prmt')
          if ~isfield(mtrcs{k}, 'prmt')
            mtrcs{k}.args.prmt  = struct();
          end
          if isfield(opts.prmt, 'PL_mode') && ~isfield(mtrcs{k}, 'PL_mode')
            mtrcs{k}.args.prmt.PL_mode = opts.prmt.PL_mode;
          end
        end
      end
      
      mtx = SensingMatrixConcat.constructConcat(mtrcs, opts);
    end
    
    function mtx = constructConcat(mtrcs, opts)
      
      function elmnts = get_elmnts(mt)
        if iscell(mt)
          elmnts = cell(1,length(mt));
          for i=1:length(mt)
            elmnts{i} = get_elmnts(mt{i});
          end
          elmnts = horzcat(elmnts{:});
          return
        elseif isstruct(mt)
          elmnts = cell(size(mt));
          for i=1:numel(mt)
            elmnts{i} = mt(i);
          end
          return
        elseif ~isa(mt,'SensingMatrixConcat') || mt.isTransposed()
            elmnts = {mt.copy()};
            return
        end
        
        % mt is a SensingMatrixConcat matrix, not transposed
        elmnts = cell(1,length(mt.mtrx));
        for i=1:length(elmnts)
          elmnt = mt.mtrx{i}.copy();
          elmnts{i} = get_elmnts(elmnt);
        end
        elmnts = horzcat(elmnts{:});
      end
      
      if nargin < 2
        opts = struct();
      end
      mtxs = get_elmnts(mtrcs);
      if numel(mtxs) > 1
        mtx = SensingMatrixConcat(mtxs, opts);
      else
        mtx = mtxs;
      end
    end
  end
end

