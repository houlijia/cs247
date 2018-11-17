classdef SensingMatrix < CompMode
  %SensingMatrix - an abstract class describing a sensing matrix
  
  properties (Constant)
    % If check_mult_eps is not empty it should be a resal non-negaive.
    % In that case the actual matrix is generated and each multVec()
    % and multVecTrnsp() checks result against actual matrix
    % multiplication. differences of magnitude exceeding check_mult_eps
    % are reported.
    % Note: This can be a huge memory guzzler!
    check_mult_eps = [];
%     check_mult_eps = 1e-8;

    % If check_gpu_epsi s not empty it should be a real non-negaive.
    % In that case before multiplying using a GPU, a multiplication is done
    % without the GPU and the two results are compared. If the norm of the
    % error exceeds check_gpu_eps times the norm of the vector, a warning
    % is issued.
    check_gpu_eps = [];
    % check_gpu_eps = 1E-10;

   % If true check every SVD computation
   chk_svd = false;
  end
  
  properties
    n_rows=0;    % Number of rows
    n_cols=0;    % Number of columns
    
    % Zeroed rows can be set so that multLR treats the matrix as if the
    % specified rows have been zeroed.  This does not apply to multVec()
    % and multTrnspVec.
    zeroed_rows=[];
    
  end
  
  properties (Access=protected)
    % If true treat the matrix as the transpose of the original matrix
    is_transposed = false;
    is_sparse = false;
    
    matrix = [];
    msrs_normalizer = [];
    msrs_normalizer_trnsp = [];
   
    % Largest singular value of the matrix. -1=undefined yet
    nrm = -1;  
    
    % Smallest singular value of the matrix. -1=undefined yet
    nrm_inv = -1;  % -1 indicates unknown yet
    exct_nrm = [false, false];
    
    % Flags indicating whether the matrix has orthonormal columns (A'*A=cI,
    % where c is a scalar), or orthonormal rows (A*A'=cI).  If there is
    % orthogonality, the value of c is nrm.
    ortho = struct('col', false, 'row', false);
    
    % Flags indicating whether the matrix has orthogonal columns (A'*A is
    % diagonal), or orthogonal rows (A*A' is diagonal).
     psd_ortho = struct('col', false, 'row', false);
     
  end
  
  properties (Access = private)
    % Lists of indices for no-clip sorting. A value of -1 means that the list
    % has not been computed yet. An empty value means that sorting (and
    % unsorting) a unit permutation (no change). Otherwise, this is the
    % prmutation, that is, if y is the measurements vector then 
    % y(sort_no_clip_indcs) is the vector sorted so that the clipped indices are
    % first and if yt is the result of the transpose of the matrix multipiled by
    % a vector then yt(sort_no_clip_trnsp) is the sorted vector with the no clip
    % element in the beginning..
    sort_no_clip_indcs = -1;
    sort_no_clip_indcs_trnsp = -1;

    % Number of no clip entries of the matrix and the transposed matrix
    n_no_clip_fwd = 0;
    n_no_clip_trnsp = 0;
    
    % No clip indices
    indcs_no_clip_fwd = [];
    indcs_no_clip_trnsp = []
  end
  
  methods (Access=protected)
    function setSensingMatrix(obj, num_rows, num_columns)
      % Input:
      %   1 argument - do nothing
      %   2 argument - seocnd argument is a struct with fields specifying the 
      %     matrix. fields are:
      %       n_cols: (required) number of columns.
      %       n_rows: (required) number of rows
      %       transpose: (optional) If present and true, the matrix is transposed.
      %     Other fields are ignored
      %  3 arguments: assumed to be num_rows and num_cols.
      switch nargin
        case 1
          return
        case 2
          opts = num_rows;
        case 3
          opts = struct('n_rows', num_rows, 'n_cols', num_columns);
        otherwise
          error('too many arguments');
      end
      
      obj.n_rows = SensingMatrix.toCPUIndex(opts.n_rows);
      obj.n_cols = SensingMatrix.toCPUIndex(opts.n_cols);
      
      if isfield(opts, 'transpose') && opts.transpose
        obj.transpose();
      end
      
      obj.setCastIndex();
      obj.setCastFloat();
    end
      
    function mtx = create(obj, args)
      obj.set(args);
      mtx = obj;
    end
    
    % Check multiplication using Matlab operations and report if an error
    % occured
    %   Input:
    %     obj - this object
    %     input - the input vector or matrix
    %     output - the computed output
    %     func_name - name of calling function
    %     mtrx - the matrix to multiply with
    %     scl - (optional) factor to scale the output by.
    %     
    function checkResult(obj, input, output, func_name, mtrx, scl)
      if ~isequal(size(output), [size(mtrx,1) size(input,2)])
        error(['SensingMatrix:' func_name ':DimensionMismtch'],...
           '%s is of size %s while output is of size %s', class(obj),...
           show_str(size(mtrx)), show_str(size(output)));
      elseif isempty(output)
        return;
      end
      otpt= mtrx*input;
      if nargin >= 6
      
        otpt = otpt*scl;
      end
      err = norm(otpt(:)-output(:), inf);
      if err > obj.check_mult_eps
        warning(['SensingMatrix:' func_name ':Inaccuracy'],...
          'SensingMatrix.%s: Error of %.3g in multiplying by %s', ...
          func_name,err, class(obj));
      end
    end
    
    function setOrtho_col(obj, val)
      if val && obj.n_cols > obj.n_rows
          error('A %d X %d matrix cannot be column orthogonal', ...
            obj.n_rows, obj.n_cols);
      end
      val = SensingMatrix.toCPULogical(val);
      if obj.n_cols == obj.n_rows
        obj.ortho.col = val;
        obj.ortho.row = val;
        if val
          obj.psd_ortho.col = val;
          obj.psd_ortho.row = val;
        end
      else
        obj.ortho.col = val;
        if val
          obj.psd_ortho.col = val;
        end
      end
    end
    
    function setOrtho_row(obj, val)
      if val && obj.n_rows > obj.n_cols
          error('A %d X %d matrix cannot be column orthogonal', ...
            obj.n_rows, obj.n_cols);
      end
      val = SensingMatrix.toCPULogical(val);
      if obj.n_cols == obj.n_rows
        obj.ortho.col = val;
        obj.ortho.row = val;
        if val
          obj.psd_ortho.col = val;
          obj.psd_ortho.row = val;
        end
      else
        obj.ortho.row = val;
        if val
          obj.psd_ortho.row = val;
        end
      end
    end
    
    function setPsdOrtho_col(obj, val)
      val = SensingMatrix.toCPULogical(val);
      obj.psd_ortho.col = val;
      if ~val
        obj.setOrtho_col(val);
      end
    end
    
    function setPsdOrtho_row(obj, val)
      val = SensingMatrix.toCPULogical(val);
      obj.psd_ortho.row = val;
      if ~val
        obj.setOrtho_row(val);
      end
    end
    
  end
  
  methods
    function obj=SensingMatrix(varargin)
      % Constructor.
      %   0 arguments - default constructor
      %   1 argument - struct with fields specifying the matrix. fields are:
      %     n_cols: (required) number of columns.
      %     n_rows: (required) number of rows
      %     transpose: (optional) If present and true, the matrix is transposed.
      %     Other fields are ignored
      %  2 arguments: assumed to be n_rows and num_cols.
      
      obj.setSensingMatrix(varargin{:});
    end
    
    function set(obj, varargin)
      % Input:
      %   0 arguments - default constructor
      %   1 argument - struct with fields specifying the matrix. fields are:
      %     n_cols: (required) number of columns.
      %     n_rows: (required) number of rows
      %     transpose: (optional) If present and true, the matrix is transposed.
      %     Other fields are ignored
      %  2 arguments: assumed to be num_rows and num_cols.
      obj.setSensingMatrix(varargin{:});
    end
    
    function eql = isEqual(obj, other)
      % This is an exact copy of the same function in CodeElement. It is
      % copied here to get access to protected properties
      if ~strcmp(class(obj), class(other)) || ...
          ~isequal(size(obj),size(other))
        eql = false;
        return;
      elseif all(eq(obj,other))
        eql = true;
        return
      end
      
      ignore_list = obj.ignoreInEqual();
      mc = metaclass(obj);
      props = mc.PropertyList;
      for k=1:length(props)
        if props(k).Constant || strcmp(props(k).GetAccess,'private') || ...
            any(strcmp(props(k).Name, ignore_list))
          continue;
        end
        prop = props(k).Name;
        
        try
          if ~isEqual(obj.(prop), other.(prop))
            eql = false;
            return;
          end
        catch exc
          if strcmp(exc.identifier, 'MATLAB:class:GetProhibited')
            continue;
          else
            rethrow(exc);
          end
        end
        
      end
      
      eql = true;
    end
        
    function transpose(obj)
      obj.is_transposed = ~obj.is_transposed;
    end
    
    % Return true if the matrix is transposed
    function is_trnsp = isTransposed(obj)
      is_trnsp = obj.is_transposed;
    end
    
    function ncl = nCols(obj)
      if obj.is_transposed
        ncl = obj.n_rows;
      else
        ncl = obj.n_cols;
      end
    end
    
    function nrw = nRows(obj)
      if obj.is_transposed
        nrw = obj.n_cols;
      else
        nrw = obj.n_rows;
      end
    end
    
    function val = isDiagonal(obj)
      val = obj.toLogical(false);
    end
    
    function orth = getOrthoCol(obj)
      if ~obj.is_transposed
        orth = obj.ortho.col;
      else
        orth = obj.ortho.row;
      end
    end
    
    function orth = getOrthoRow(obj)
      if ~obj.is_transposed
        orth = obj.ortho.row;
      else
        orth = obj.ortho.col;
      end
    end
    
    function orth = getOrtho(obj)
      orth = obj.ortho.col && obj.ortho.row;
    end
    
    function setOrthoCol(obj, val)
      val = SensingMatrix.toCPULogical(val);
      if ~obj.is_transposed
        obj.setOrtho_col(val);
      else
        obj.setOrtho_row(val);
      end
    end
    
    function setOrthoRow(obj, val)
      val = SensingMatrix.toCPULogical(val);
      if ~obj.is_transposed
        obj.setOrtho_row(val);
      else
        obj.setOrtho_col(val);
      end
    end
    
    function setOrtho(obj, val)
      val = SensingMatrix.toCPULogical(val);
      obj.setOrtho_col(val);
      obj.setOrtho_row(val);
    end
    
    function orth = getPsdOrthoCol(obj)
      if ~obj.is_transposed
        orth = obj.psd_ortho.col;
      else
        orth = obj.psd_ortho.row;
      end
    end
    
    function orth = getPsdOrthoRow(obj)
      if ~obj.is_transposed
        orth = obj.psd_ortho.row;
      else
        orth = obj.psd_ortho.col;
      end
    end
    
    function setPsdOrthoCol(obj, val)
      val = SensingMatrix.toCPULogical(val);
      if ~obj.is_transposed
        obj.setPsdOrtho_col(val);
      else
        obj.setPsdOrtho_row(val);
      end
    end
    
    function setPsdOrthoRow(obj, val)
      val = SensingMatrix.toCPULogical(val);
      if ~obj.is_transposed
        obj.setPsdOrtho_row(val);
      else
        obj.setPsdOrtho_col(val);
      end
    end
    

    % Set the zeroed rows for multLR
    function setZeroedRows(obj, zr)
      obj.zeroed_rows = obj.toIndex(zr);
      obj.clearMatrix();
    end
    
%     function eql = isEqual(obj, other)
%       if class(obj) ~= class(other)
%         eql = false;
%         return
%       end
%       
%       otr = other.copy();
%       otr.use_gpu = obj.use_gpu;
%       otr.use_single = obj.use_single;
%       otr.matrix = obj.matrix;
%       otr.nrm = obj.nrm;
%       otr.nrm_inv = obj.nrm_inv;
%       otr.exct_nrm = obj.exct_nrm;
%       eql = isEqual@CodeElement(obj,otr);
%     end
    
    % Clear the computed matrix
    function clearMatrix(obj)
      obj.matrix = obj.toFloat([]);
    end
    
    % multVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = multVec(obj, x)
%       x = obj.toFloat(x);
      if ~isempty(SensingMatrix.check_gpu_eps) && isa(x, 'gpuArray')
        yref = obj.multVec(gather(x));
      else
        yref = [];
      end
      
      if obj.is_transposed
        x(obj.zeroed_rows) = 0;
        y = obj.doMultTrnspVec(x);
      else
        y = obj.doMultVec(x);
        y(obj.zeroed_rows) = 0;
      end
      
      if ~isempty(yref)
        dff = norm(yref-y);
        rt = dff/(norm(x)+1E-10);
        if rt >= SensingMatrix.check_gpu_eps
          warning('GPU result different from CPU result. norm_ratio=%g',rt);
        end
      end
      
      if ~isempty(obj.check_mult_eps)
        obj.checkResult(x, y, 'multVec', obj.getMatrix());
      end
    end
    
    % multTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = multTrnspVec(obj, x)
%       x = obj.toFloat(x);
      if ~isempty(SensingMatrix.check_gpu_eps) && isa(x, 'gpuArray')
        yref = obj.multTrnspVec(gather(x));
      else
        yref = [];
      end
      
      if obj.is_transposed
        y = obj.doMultVec(x);
        y(obj.zeroed_rows) = 0;
      else
        x(obj.zeroed_rows) = 0;
        y = obj.doMultTrnspVec(x);
      end
      
      if ~isempty(yref)
        dff = norm(yref-y);
        rt = dff/(norm(x)+1E-10);
        if rt >= SensingMatrix.check_gpu_eps
          warning('GPU result different from CPU result. norm_ratio=%g',rt);
        end
      end
      
      if ~isempty(obj.check_mult_eps)
        obj.checkResult(x, y, 'multTrnspVec', obj.getMatrix()');
      end
    end
    
    % multLR chooses invokes multVec if mode is false and multTrnspVec  if mode
    % is true.
    function y = multLR(obj, x, mode)
%       x = obj.toFloat(x);
      if mode
        y = obj.multTrnspVec(x);
      else
        y = obj.multVec(x);
      end
    end
    
    % Return a function handle for multLR
    function hndl = getHandle_multLR(obj)
      hndl = @(x,mode) multLR(obj, x, mode);
    end
    
    function y = multMat(obj, x)
%       x = obj.toFloat(x);
      if ~isempty(SensingMatrix.check_gpu_eps) && isa(x, 'gpuArray')
        yref = obj.multMat(gather(x));
      else
        yref = [];
      end
      
      if obj.is_transposed
        y = obj.doMultTrnspMat(x);
        y(obj.zeroed_rows,:) = 0;
      else
        x(obj.zeroed_rows,:) = 0;
        y = obj.doMultMat(x);
      end
      
      if ~isempty(yref)
        dff = norm(yref-y);
        rt = dff/(norm(x)+1E-10);
        if rt >= SensingMatrix.check_gpu_eps
          warning('GPU result different from CPU result. norm_ratio=%g',rt);
        end
      end
      
      if ~isempty(obj.check_mult_eps)
        obj.checkResult(x, y, 'multMat', obj.getMatrix());
      end
    end
    
    function y = multTrnspMat(obj, x)
%       x = obj.toFloat(x);
      if ~isempty(SensingMatrix.check_gpu_eps) && isa(x, 'gpuArray')
        yref = obj.multTrnspMat(gather(x));
      else
        yref = [];
      end
      
      if obj.is_transposed
        y = obj.doMultMat(x);
        y(obj.zeroed_rows,:) = 0;
      else
        x(obj.zeroed_rows,:) = 0;
        y = obj.doMultTrnspMat(x);
      end
      
      if ~isempty(yref)
        dff = norm(yref-y);
        rt = dff/(norm(x)+1E-10);
        if rt >= SensingMatrix.check_gpu_eps
          warning('GPU result different from CPU result. norm_ratio=%g',rt);
        end
      end
      
      if ~isempty(obj.check_mult_eps)
        obj.checkResult(x, y, 'multTrnspMat', obj.getMatrix()');
      end
    end
    
    function y = doMultMat(obj, x)
      if isa(x,'gpuArray')
        rw = gpuArray(1:size(x,2));
      else
        rw = (1:size(x,2));
      end
      yc = arrayfun(@(k) {obj.doMultVec(x(:,k))}, rw);
      y = horzcat(yc{:});
%       y = zeros(obj.n_rows, size(x,2));
%       y = zeros(obj.n_rows, size(x,2));
%       for k=1:size(x,2)
%         y(:,k) = obj.doMultVec(x(:,k));
%       end
    end
    
    function y = doMultTrnspMat(obj, x)
      if isa(x,'gpuArray')
        rw = gpuArray(1:size(x,2));
      else
        rw = (1:size(x,2));
      end
      yc = arrayfun(@(k) {obj.doMultTrnspVec(x(:,k))}, rw);
      y = horzcat(yc{:});
%       y = zeros(obj.n_cols, size(x,2));
%       for k=1:size(x,2)
%         y(:,k) = obj.doMultTrnspVec(x(:,k));
%       end
    end
    
    % Multiply a matrix or a vector by the matrix whose entries are the
    % absolute values of the entries of this matrix.
    function y = multAbs(obj, x)
      x = obj.toFloat(x);
      m = abs(obj.getMatrix());
      y = m * x;
    end
    
    % Multiply a matrix or a vector by the matrix whose entries are the
    % absolute values of the entries of the transpose of this matrix.
    function y = multTrnspAbs(obj, x)
      x = obj.toFloat(x);
      obj.transpose();
      y = obj.multAbs(x);
      obj.transpose();
    end      
      
    function mtrx = getMatrix(obj, keep_mtx)
      % Get a Matlab matrix which is the same as this matrix (including
      % transpose effect). If keep_mtx exists and is true, the matrix is
      % stored for future calls.
      if ~obj.is_transposed
        if isempty(obj.matrix)
          mtrx = obj.compMatrix();
          if nargin > 1 && keep_mtx
            obj.matrix = mtrx;
          end
        else
          mtrx = obj.matrix;
        end
      else
        if isempty(obj.matrix)
          mtrx = obj.compMatrix();
          if nargin > 1 && keep_mtx
            obj.matrix = mtrx';
          end
        else
          mtrx = obj.matrix';
        end
      end
    end
    
    % Compute the Matlab matrix idential to this matrix, ignoring the
    % transpose
    function mtx = compMatrix(obj)
      mtx = obj.doCompMatrix();
      if ~obj.is_transposed
        mtx(obj.zeroed_rows,:) = ...
          builtin('zeros',length(obj.zeroed_rows),size(mtx,2), 'like', mtx);
      else
        mtx(:,obj.zeroed_rows) = ...
          builtin('zeros',size(mtx,1),length(obj.zeroed_rows), 'like', mtx);
      end
    end
    
    % Compute the Matlab matrix idential to this matrix, but without
    % zeroing rows
    function mtx = doCompMatrix(obj)
      nc = obj.nCols();
      nr = obj.nRows();
      if nc <= nr
        mtx = obj.multMat(obj.toFloat(eye(nc)));
      else
        mtx = obj.multTrnspMat(obj.toFloat(eye(nr)));
        mtx = mtx';
      end
    end
    
    % returns true if getMatrix returns a sparse matrix
    function is_sprs = isSparse(obj)
      is_sprs = obj.is_sparse;
    end
    
    function setSparse(obj,val)
      obj.is_sparse = logical(val);
    end
    
    function setMatrix(obj, mtrx)
      obj.matrix = obj.toFloat(mtrx);
      if ~obj.use_gpu && obj.is_sparse
        obj.matrix = sparse(obj.matrix);
      end
    end
    
    % Some sensing matrices have some special rows: The output from
    % these rows may have a special significance and should not be
    % clipped in the quantizer, e.g. if that output is the DC value
    % The following functions support this capability
    
    function out = sortNoClip(obj, y)
      % Sort the measurements vector y so that the no clip elements are first.
      indcs = obj.cmpSortNoClip();
      if isempty(indcs)
        out = y;
      else
        out = y(indcs);
      end
    end
    
    function out = unsortNoClip(obj, y)
      % Unsort the sorted vector y so that the no clip elements are in
      % their original place.
      indcs = obj.cmpSortNoClip();
      if isempty(indcs)
        out = y;
      else
        out = builtin('zeros', size(y), 'like', y);
        out(indcs) = y;
      end
    end
    
    function n_no_clip = nNoClip(obj, trnsp)
      % Return the number of no-clip elements in the output vector, 
      % according to the trnaspsoe state of the matrix. If trnsp is present
      % and true the result is returned according to reverse transpose state of
      % the matrix
      
      if nargin < 2
        trnsp = false;
      end

      if xor(obj.isTransposed(), trnsp)
        n_no_clip = obj.n_no_clip_trnsp;
      else
        n_no_clip = obj.n_no_clip_fwd;
      end
    end
    
    function indcs = indcsNoClip(obj, trnsp)
      % Return the indices of the no-clip elements as an ascending (column)
      % list, according to the trnaspsoe state of the matrix. If trnsp is present
      % and true the result is returned according to reverse transpose state of
      % the matrix
      
      if nargin < 2
        trnsp = false;
      end

      if xor(obj.isTransposed(), trnsp)
        indcs = obj.toCPUIndex(obj.indcs_no_clip_trnsp);
      else
        indcs = obj.toCPUIndex(obj.indcs_no_clip_fwd);
      end
    end
    
    function setIndcsNoClip(obj, indcs_no_clip, trnsp)
      % Set the no-clip indics to be indcs_no_clip (for the current
      % non-transpose or transpos state of obj). If trnsp is present
      % and true the result is returned according to reverse transpose state of
      % the matrix
      
      if nargin < 3
        trnsp = false;
      end

      nn = length(indcs_no_clip);
      indcs_no_clip = sort(indcs_no_clip);
      
      if xor(obj.isTransposed(), trnsp)
        obj.sort_no_clip_indcs_trnsp = -1;
        obj.indcs_no_clip_trnsp = indcs_no_clip;
        obj.n_no_clip_trnsp = nn;
      else
        obj.sort_no_clip_indcs = -1;
        obj.indcs_no_clip_fwd = indcs_no_clip;
        obj.n_no_clip_fwd = nn;
      end
        
    end
    
    function mtrx = cmpSortNoClipMtrx(obj, mark_no_clip)
      % cmsSortNoClipMtrx returns a SensingMatrix which performs the sorting on 
      % the measurements produced by this object.
      %   INPUT:
      %     obj: this object
      %     mark_no_clip: (optional) if present and true, mark the first
      %       obj.nNoClip() entries in the output as no clip
      %   OUTPUT:
      %     mtrx: the output matrix
      
      if nargin < 2
        mark_no_clip = false;
      end
      
      nc = obj.nRows();
      indcs = obj.cmpSortNoClip();
      if isempty(indcs)
        mtrx = SensingMatrixUnit(nc);
      else
        mtrx = SensingMatrixSelect.construct(indcs, nc);
      end
      
      if mark_no_clip
        mtrx.setNoClip(1:obj.nNoClip());
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
    %                            ofst_msrs (see below)
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
    function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts,~, ~,params)
      lofst = length(ofsts);
      ofst_msrs = obj.zeros(0,lofst);
      
      if nargin >= 5 && ...
          isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
        ofst_msrs  = obj.getEdgeMsrmnts(ofst_msrs, params.ofsts_list,...
          params.nghbr_list);
      end
      
    end
    
    % Returns the sum of values of the measurement which contain DC value,
    % weighted by the ratio of the DC value to other components (in
    % terms of RMS), or 0 if there is no such measurement.
    %   Input:
    %     obj - this object
    %     msrs - the measurements vector
    function dc_val = getDC(obj,~)
      dc_val = obj.toFloat(0);  %default when DC is not known
    end
    
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(obj,y)
      y = obj.toFloat(y);
      if obj.is_transposed
        if isempty(obj.msrs_normalizer_trnsp)
          obj.msrs_normalizer_trnsp = obj.compMsrsNormalizer();
        end
        y = y ./ obj.msrs_normalizer_trnsp;
      else
        if isempty(obj.msrs_normalizer)
          obj.msrs_normalizer = obj.compMsrsNormalizer();
        end
        y = y ./ obj.msrs_normalizer;
      end
    end
    
    % undo the operation of normalizeMsrs
    function y = deNormalizeMsrs(obj,y)
      y = obj.toFloat(y);
      if obj.is_transposed
        if isempty(obj.msrs_normalizer_trnsp)
          obj.msrs_normalizer_trnsp = obj.compMsrsNormalizer();
        end
        y = y .* obj.msrs_normalizer_trnsp;
      else
        if isempty(obj.msrs_normalizer)
          obj.msrs_normalizer = obj.compMsrsNormalizer();
        end
        y = y .* obj.msrs_normalizer;
      end
    end
    
    % Returns the square of the L2 matrix norm of A which is the largest 
    % eigenvalue of A'A. This is the square of maximum of
    % norm(Ax,2) for any x such that norm(x,2)==1. In some cases the value
    % given by this function is not exact but an upper bound.
    function y = norm(obj)
      if obj.nrm == -1
        obj.nrm = obj.toCPUFloat(obj.compNorm());
      end
      y = obj.nrm;
    end
    
    function setNorm(obj, val, is_exact)
      % Set a value for Norm(). To clear the current value set [] or -1.
      % If val is a SensingMatrix of the same type, norms are copied.
      if isempty(val)
        obj.nrm(1) = obj.toCPUFloat(-1);
        obj.exct_nrm(1) = SensingMatrix.toCPULogical(false);
        return
      end
      if isa(val, class(obj))
        obj.nrm = obj.toCPUFloat(val.nrm);
        obj.nrm_inv = obj.toCPUFloat(val.nrm_inv);
        obj.exct_nrm = SensingMatrix.toCPULogical(val.exct_nrm);
      else
        if nargin < 3
          is_exact = false;
        end
        if obj.nrm >= 0 && obj.exct_nrm(1) && ~is_exact 
          return; % we already have a better value
        end
        obj.nrm = obj.toCPUFloat(val);
        obj.exct_nrm(1) = SensingMatrix.toCPULogical(is_exact);
      end
      
    end
    
    function val = getExactNorm(obj)
      % Set an exact value for norm(). It can be computationally heavy
      if obj.exct_nrm(1)
        val = obj.nrm;
      else
        val = obj.toCPUFloat(obj.cmpExactNorm());
        obj.setNorm(val, true);
      end
    end
    
    % Set an exact value for norm(). It can be computationally heavy
    function val = cmpExactNorm(obj)
      o_pcol = obj.getPsdOrthoCol();
      o_prow = obj.getPsdOrthoRow();
      o_diag = obj.isDiagonal();
      nc = obj.nCols();
      nr = obj.nRows();
      if o_diag || o_pcol || o_prow
        if o_diag
          dg = obj.getDiag();
          dg = abs(dg);
        elseif o_pcol
          if obj.getOrthoCol()
            dg = colEntryNrm(1);
          else
            dg = arrayfun(@colEntryNrm, 1:nc);
%             dg = zeros(obj.nRows(),1);
%             nc = obj.nCols();
%             for k=1:length(dg)
%               x = zeros(nc,1);
%               x(k) = 1;
%               dg(k) = norm(obj.multVec(x));
%             end
          end
        else   % prow
          if obj.getOrthoRow()
            dg = rowEntryNrm(1);
%             dg = norm(obj.multTrnspVec([1;zeros(obj.nRows()-1,1)]));
          else
            dg = arrayfun(@rowEntryNrm, 1:nr);
%             dg = zeros(obj.nCols(),1);
%             nr = obj.nRows();
%             for k=1:length(dg)
%               x = zeros(nr,1);
%               x(k) = 1;
%               dg(k) = norm(obj.multTrnspVec(x));
%             end
          end
        end
        val = max(dg(:));
      elseif obj.isSparse()
        % GPU does not support sparse matrices
        val = svds(gather(obj.getMatrix()),1);
      else
        val = norm(full(obj.getMatrix()));
      end
      
      function val = colEntryNrm(k)
        x =obj.zeros(nc,1);
        x(k) = 1;
        val = norm(obj.multVec(x));
      end
        
      function val = rowEntryNrm(k)
        x =obj.zeros(nr,1);
        x(k) = 1;
        val = norm(obj.multTrnspVec(x));
      end
        
    end
    
    function msrs_noise= calcMsrsNoise(obj, n_orig_pxls, pxl_width)
      % Compute standard deviation of noise in measurement, assuming that
      % the noise results from pixel quantization
      %   Input:
      %     n_orig_pxls - number of original pixels
      %     pxl_width - width of pixel quantization interval
      %                 (optional, default = 1 for uniform)
      
      n_pxls = obj.nCols();
      mtrx_nrm = obj.norm();
      if nargin < 3
        pxl_width = 1;
        if nargin < 2
          n_orig_pxls = n_pxls;
        end
      end
      msrs_noise = mtrx_nrm * pxl_width * sqrt(double(n_pxls)/double(12*n_orig_pxls));
    end
     
    % Compute the SVD of the matrix: A=U*L*V', where U and L are
    % column-orthogonal and L is diagonal. If eps is specified call
    % truncdate SVD afterwards.
    %   Input:
    %     obj - this object, the matrix of which SVD is computed.
    %     eps - error tolerance. Optional, Default=0
    %     complete - If true a complete SVD is computed, thus U and V are
    %                square matrices and L is a diagonal rectangular
    %                matrix. No truncation is done in this case. Optional,
    %                default = flase
    %  Output arguments:
    %    components of the SVD, such that this matrix is U*L*V', U,V have
    %    orthonormal columns and L is diagonal
    function [L,U,V] = compSVD(obj, eps, complete)
      if nargin < 3
        complete = false;
        if nargin < 2
          eps = 0;
        end
      end
      
      eps =obj.toFloat(eps);
      
      do_sort = obj.toLogical(eps > 0);
      if nargout == 1
        if isa(obj,'SensingMatrixSelect')
          L = obj.do_compSVD3(complete);
        elseif obj.isDiagonal()
          L = obj.copy();
          do_sort = true;
        elseif obj.getOrthoCol() || obj.getOrthoRow()
          scl = obj.getExactNorm();
          if ~complete
            L = SensingMatrixScaler.construct(scl, min(obj.n_rows, obj.n_cols));
          else
            L = SensingMatrixScaler.construct(scl, obj.nRows(), obj.nCols());
          end
        elseif obj.getPsdOrthoCol()
          mtx = obj.getMatrix();
          dg = sort(sqrt(diag(mtx'*mtx)), 'descend');
          if ~complete
            L = SensingMatrixDiag.constructDiag(dg);
          else
            L = SensingMatrixDiag.constructDiag(dg, obj.nRows(), obj.nCols());
          end
        elseif obj.getPsdOrthoRow()
          mtx = obj.getMatrix();
          dg = sort(sqrt(diag(mtx*mtx')), 'descend');
          if ~complete
            L = SensingMatrixDiag.constructDiag(dg);
          else
            L = SensingMatrixDiag.constructDiag(dg, obj.nRows(), obj.nCols());
          end
%           obj.transpose();
%           L = obj.compSVD(eps, complete);
%           obj.transpose();
%           L = obj.transposeSVD(L);
        else
          L = obj.do_compSVD1(complete);
        end
        if do_sort
          L = obj.sortSVD(L, [], [], eps);
        end
        if ~complete
          L = obj.truncateSVD(L,[],[]);
        end
        obj.chkSVD(L,[],[],2*eps, complete);
      else % nargout == 3
        if isa(obj,'SensingMatrixSelect')
          [L,U,V] = obj.do_compSVD3(complete);
        elseif obj.isDiagonal()
          if complete
            L = obj.copy();
            U = SensingMatrixUnit(obj.nRows());
            V = SensingMatrixUnit(obj.nCols());
          else
            nr = obj.nRows();
            nc = obj.nCols();
            L = SensingMatrixDiag.constructDiag(obj.getDiag());
            if nr >= nc
              U = SensingMatrixSelectRange.constructSelectRange(1,nc,nr);
              U.transpose();
              V = SensingMatrixUnit(nc);
            else
              U = SensingMatrixUnit(nr);
              V = SensingMatrixSelectRange.constructSelectRange(1,nr,nc);
              V.transpose();
            end
          end
          do_sort = true;
        elseif obj.getPsdOrthoCol() && (~complete || obj.getPsdOrthoRow())
          if obj.getOrthoCol() && (~complete || obj.getOrthoRow())
            scl = sqrt(obj.getExactNorm());
            if scl == 1
              L = SensingMatrixUnit(obj.nCols());
            else
              L = SensingMatrixScaler(obj.nCols(),scl);
            end
            if scl >= eps
              U = SensingMatrixCascade.constructCascade({...
                obj, SensingMatrixScaler(obj.nRows(),1/scl)});
            else
              U=SensingMatrixUnit(obj.nRows());
            end
            V = SensingMatrixUnit(obj.nCols());
          else
            mtx = obj.getMatrix();
            [dg,ordr] = sort(sqrt(diag(mtx'*mtx)), 'descend');
            mtx = mtx(:,ordr);
            indx = find(dg>0);
            mtx(:,indx) = mtx(:,indx) ./ ...
              (builtin('ones',size(mtx,1),1,'like',mtx)* dg(indx)');
            U = SensingMatrixMatlab(mtx);
            U.setOrthoCol(true);
            L = SensingMatrixDiag(dg);
            V = SensingMatrixSelect(ordr,obj.nCols());
            V.transpose();
          end
        elseif obj.getPsdOrthoRow() && ~complete
          obj.transpose();
          [L,U,V] = obj.compSVD(eps, complete);
          obj.transpose();
          [L,U,V] = obj.transposeSVD(L,U,V);
        else
          [L,U,V] = obj.do_compSVD3(complete);
        end
        if do_sort
          [L,U,V] = obj.sortSVD(L, U, V, eps);
        end
        if ~complete
          [L,U,V] = obj.truncateSVD(L, U, V);
        end
        obj.chkSVD(L,U,V,2*eps,obj, complete);

        U.use_gpu = obj.use_gpu;        
        U.use_single = obj.use_single;
        V.use_gpu = obj.use_gpu;        
        V.use_single = obj.use_single;
      end
      L.use_gpu = obj.use_gpu;
      L.use_single = obj.use_single;
    end
    
    function L = do_compSVD1(obj, complete)
      nr = obj.nRows();
      nc = obj.nCols();
      L = svd(full(obj.getMatrix()));
      if complete
        L = SensingMatrixDiag.constructDiag(diag(L), nr, nc);
      else
        L = SensingMatrixDiag.constructDiag(diag(L));
      end
    end
    
    function [L,U,V] = do_compSVD3(obj,complete) 
      nr = obj.nRows();
      nc = obj.nCols();
      if ~complete
          [U,L,V] = svd(full(obj.getMatrix()),'econ');
        L = SensingMatrixDiag.constructDiag(diag(L));
      else
        [U,L,V] = svd(full(obj.getMatrix()));
        L = SensingMatrixDiag.constructDiag(diag(L), nr, nc);
      end
      U = SensingMatrixMatlab(U);
      U.setOrthoCol(true);
      V = SensingMatrixMatlab(V);
      V.setOrthoCol(true);
    end
    
  end
  
  methods(Abstract)
    % doMultVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    y = doMultVec(obj, x)
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    y = doMultTrnspVec(obj, x)
    
  end
  
  methods (Static)
    function mtrx = construct(type,args)
      % Generate matrix of a given type and arguments
      try
        mtrx = eval(type);
      catch excpt
        if isempty(regexp(type, '^SensingMatrix', 'once'))
          type = ['SensingMatrix', type];
          mtrx = eval(type);
        else
          excpt.rethrow();
        end
      end
      
      mtrx = mtrx.create(args);
    end
    
    % MultOrient - multiply vector x by SensingMatrix, matrix. If orientation
    % is 1, multiply by the matrix itsself; if it is 2 multiply by the
    % transpose. return output in y.
    function y = multOrient(matrix, x, orientation)
      if orientation == 1
        y = matrix.multVec(x);
      else
        y = matrix.multTransposeVec(x);
      end
    end
    
    % Discard all zero singular values whose values
    % and discard the corresponding singular vectors. The
    % SVD components are U,L,V, where U and V are column orthogonal and L
    % is diagonal.
    function [L,U,V] = truncateSVD(L, U, V)
      if nargin < 3 || nargout < 3
        U = [];
        if nargin < 2 || nargout < 2
          V=[];
        end
      end
      dg = L.getDiag();
      indx = find (dg == 0, 1);
      if isempty(indx)
        return;
      end
      
      L = SensingMatrixDiag.constructDiag(dg(1:indx-1));
      if ~isempty(U)
        U = remove_cols(U);
        if ~isempty(V)
          V = remove_cols(V);
        end
      end
      
      function S = remove_cols(S)
        if isa(S,'SensingMatrixMatlab')
          m = S.getMatrix();
          m = m(:,1:(indx-1));
          S = SensingMatrixMatlab(m);
        else
          m = SensingMatrixSelectRange.constructSelectRange(1, indx-1, S.nRows());
          m.transpose();
          S = SensingMatrixCascade.constructCascade({S,m});
        end
        S.setOrthoCol(true);
      end
    end
    
    % Sort singular values in decreasing order
    %   Input:
    %     L - A diagonal matrix with non-negative elements
    %     U,V - (optional) matrices with orthogonal columns, such that U*L*V' is well
    %           defined
    %     eps - (optional) tolerance. If present, singular values which
    %           differ by less than eps * largest singular value are set to
    %           their mean value.
    %   Output
    %     L,U,V sorted matrices
    function [L,U,V] = sortSVD(L,U,V, eps)
      dg = L.getDiag();
      [dgs,ordr]= sort(dg(:),1,'descend');
      do_vals = false;
      if nargin >= 4
        thresh = dgs(1)*eps;
        indx = find(dgs < thresh,1);
        if isempty(indx)
          k_end = length(ordr);
        else
          k_end = indx-1;
          if dgs(indx) > 0
            do_vals = true;
            dgs(indx:end) = 0;
            ordr(indx:end) = sort(ordr(indx:end));
          end
        end
        
        k=1;
        while k<k_end
          n = find(dgs(k:k_end) >= dgs(k)-thresh, 1, 'last');
          if n > 1
            n=k+n-1;
            if dgs(n) < dgs(k)
              do_vals = true;
              dgs(k:n) = mean(dgs(k:n));
              ordr(k:n) = sort(ordr(k:n));
            end
            k = n+1;
          else
            k = k+1;
          end
        end
      end
      do_sort = ~isequal(ordr(:),(1:length(ordr))');
      if do_sort || do_vals 
        L = SensingMatrixDiag.constructDiag(dgs, L.nRows(), L.nCols());
      end
      if nargout > 1 && do_sort
        nr = L.nRows();
        nc = L.nCols();
        no = length(ordr);
        P = SensingMatrixSelect([ordr(:);(no+1:nr)'], nr);
        P.transpose();
        U = SensingMatrixCascade.constructCascade({U,P});
        P = SensingMatrixSelect([ordr(:);(no+1:nc)'], nc);
        P.transpose();
        V = SensingMatrixCascade.constructCascade({V,P});
      end
    end
    
    % Check the correctness of the SVD decomposition M = U*L*V'
    % Where L is diagonal with non-negative diagonal elements in 
    % decreasing order, U,V are colunm orthonormal matrices, i.e. U'*U=I,
    % and V'*V=I. eps is an accuracy threshold. If not given it is assumed
    % to be zero. If M is not given, the function will check only if U,L,V
    % meet the condtions, but not if M=U*L*V'
    function chkSVD(L,U,V,eps,M, complete)
      if ~SensingMatrix.chk_svd
        return
      end
      
      if nargin <6
        complete = false;
        if nargin < 5
          M = [];
          if nargin < 4
            eps = 0;
            if nargin < 3
              U = [];
              V = [];
            end
          end
        end
      end
      
      % Check L
      if ~isa(L, 'SensingMatrix')
        warning('SensingMatrix:chkSVD',...
          'L is not a SensingMatrix objects');
        return
      end
      
      if ~L.isDiagonal()
        warning('SensingMatrix:chkSVD','L is not diagonal');
      end
      dg = L.getDiag();
      if ~isempty(dg)
        if any(dg(1:end-1) < dg(2:end))
          warning('SensingMatrix:chkSVD','L is not non-increasing');
        elseif dg(end)<0
          warning('SensingMatrix:chkSVD','L is not non-negative');
        end
      end
      
      if isempty(U)
        return
      end
      
      % Check dimension
      if U.nCols() ~= L.nRows()
        warning('SensingMatrix:chkSVD',...
          'U is %dx%d but L is %dx%d)', U.nRows(),U.nCols(),...
          L.nRows(), L.nCols());
        return
      end
      
      if ~isempty(V) && V.nCols() ~= L.nCols()
        warning('SensingMatrix:chkSVD',...
          'U is %dx%d but L is %dx%d)', U.nRows(),U.nCols(),...
          L.nRows(), L.nCols());
        return
      end
        
      % Check U and V
      chk_UV(U,'U');
      chk_UV(V, 'V');
      
      if isempty(M)
        return
      end
      
      % Check M
      if complete && ~isequal([L.nRows(), L.nCols()], [M.nRows(), M.nCols()])
        warning('SensingMatrix:chkSVD','L is not the same size as the matrix M');
      end

        
      m = M.getMatrix();
      ULVt = SensingMatrix.constructSVD(L,U,V);
      ulvt = ULVt.getMatrix();
      err = full(max(max(abs(m - ulvt))));
      if err>0 && err > eps*dg(1)
        warning('SensingMatrix:chkSVD','ULVt-M error is %g', err);
      end
        
      function chk_UV(x, name)
        if isempty(x)
          return
        end
        if ~isa(x,'SensingMatrix')
          warning('SensingMatrix:chkSVD',...
            '%s is not a Sensing Matrix object', name);
          return
        end
        
        if complete && x.nCols() ~= x.nRows()
          warning('SensingMatrix:chkSVD',...
            '%s is not square', name);
        end
        if ~x.getOrthoCol()
          warning('SensingMatrix:chkSVD',...
            '%s is not marked as column orthogonal', name);
        end
        xt = copy(x);
        xt.transpose();
        xtx = SensingMatrixCascade.constructCascade({xt,x});
        m = xtx.getMatrix();
        err = full(max(max(abs(m - speye(x.nCols())))));
        if err > eps
          warning('SensingMatrix:chkSVD',...
            '%s is not column orthogonal err=%g', name, err);
        end
      end
    end
    
    % Reconstruct a matrix from its SVD. M = U*L*V'
    function M = constructSVD(L,U,V)
      VC = V.copy();
      VC.transpose();
      M = SensingMatrixCascade.constructCascade({U,L,VC});
    end
    
    function [LI,UI,VI] = transposeSVD(L,U,V)
      LI = L.copy();
      LI.transpose();
      if nargin > 1
        UI = V.copy();
        VI = U.copy();
      end
    end
    
    function [LI,UI,VI] = invertSVD(L,U,V, eps)
      if nargin < 4
        eps = 0;
      end
      dg = L.getDiag();
      dg(dg<=eps) = 1;
      dg = 1 ./ dg;
      LI = SensingMatrixDiag.constructDiag(dg, L.nCols(), L.nRows());
      if nargout > 1
        UI = V.copy();
        VI = U.copy();
      end        
    end
        
  end
  
  methods (Access=protected)
    function y = compNorm(obj)
      y = getExactNorm(obj);
%       if obj.getOrthoCol()
%         x = zeros(obj.nCols(),1);
%         x(1) = 1;
%         x = obj.multVec(x);
%         y = norm(x);
%         return
%       elseif obj.getPsdOrthoCol()
%         x = ones(obj.nCols(),1);
%         x = obj.multTrnspVec(obj.multVec(x));
%         y  = sqrt(max(x));
%         return
%       end
%       m = obj.getMatrix();
%       if issparse(m)
%         y = svds(m,1);
%       else
%         y = norm(full(obj.getMatrix()));
%       end
%       y = y^2;
    end
    
    function indcs = cmpSortNoClip(obj)
      % Compute sort_no_clip_indcs or sort_no_clip_indcs_trnsp
 
      if obj.isTransposed()
        indcs = obj.sort_no_clip_indcs_trnsp;
      else
        indcs = obj.sort_no_clip_indcs;
      end
      
      if isempty(indcs) || indcs(1) > 0
        return
      end
      
      m = obj.nNoClip();
      if m == 0
        indcs = [];
      else
        indcs = obj.indcsNoClip();
        if isequal(sort(indcs), (obj.toCPUIndex(1):obj.toCPUIndex(m))')
          indcs = [];
        else
          indcs = obj.toIndex(indcs);
          c_indcs = (obj.toIndex(1):obj.toIndex(obj.nRows()))';
          c_indcs(indcs) = [];
          indcs = [indcs; c_indcs];
        end
      end
      if obj.isTransposed()
        obj.sort_no_clip_indcs_trnsp = obj.toIndex(indcs);
      else
        obj.sort_no_clip_indcs = obj.toIndex(indcs);
      end
    end
        
    function setCastIndex(obj)
      obj.setCastIndex@CompMode();
      if ~obj.use_gpu
        obj.zeroed_rows = gather(obj.zeroed_rows);
      end
      obj.is_transposed = SensingMatrix.toCPULogical(obj.is_transposed);
      obj.is_sparse = SensingMatrix.toCPULogical(obj.is_sparse);
      obj.zeroed_rows = obj.toIndex(obj.zeroed_rows);
      obj.exct_nrm = SensingMatrix.toCPULogical(obj.exct_nrm);
      obj.ortho.col = SensingMatrix.toCPULogical(obj.ortho.col);
      obj.ortho.row = SensingMatrix.toCPULogical(obj.ortho.row);
      obj.psd_ortho.col = SensingMatrix.toCPULogical(obj.psd_ortho.col);
      obj.psd_ortho.row = SensingMatrix.toCPULogical(obj.psd_ortho.row);
      
      if obj.use_gpu
        if ~isempty(obj.sort_no_clip_indcs) && obj.sort_no_clip_indcs(1) > 0
          obj.sort_no_clip_indcs = obj.toGPUIndex(obj.sort_no_clip_indcs);
        end
        if ~isempty(obj.sort_no_clip_indcs_trnsp) && ...
            obj.sort_no_clip_indcs_trnsp(1) > 0
          obj.sort_no_clip_indcs_trnsp = ...
            obj.toGPUIndex(obj.sort_no_clip_indcs_trnsp);
        end
      else
        if ~isempty(obj.sort_no_clip_indcs) && obj.sort_no_clip_indcs(1) > 0
          obj.sort_no_clip_indcs = ...
            obj.toCPUIndex(gather(obj.sort_no_clip_indcs));
        end
        if ~isempty(obj.sort_no_clip_indcs_trnsp) && ...
            obj.sort_no_clip_indcs_trnsp(1) > 0
          obj.sort_no_clip_indcs_trnsp = ...
            obj.toCPUIndex(gather(obj.sort_no_clip_indcs_trnsp));
        end
      end
      
    end
    
    function setCastFloat(obj)
      obj.setCastFloat@CompMode();
      
      if ~obj.use_gpu
        obj.matrix = gather(obj.matrix);
        obj.msrs_normalizer = gather(obj.msrs_normalizer);
        obj.msrs_normalizer_trnsp = gather(obj.msrs_normalizer_trnsp);
        obj.nrm = gather(obj.nrm);
        obj.nrm_inv = gather(obj.nrm_inv);
      end
      if ~isempty(obj.matrix)
        obj.matrix = obj.toFloat(obj.matrix);
        if ~obj.use_gpu && ~obj.use_single && obj.is_sparse
          obj.matrix = sparse(double(obj.matrix));
        end
      end
      if ~isempty(obj.msrs_normalizer) || ~isempty(obj.msrs_normalizer_trnsp)
        obj.msrs_normalizer = obj.toFloat(obj.msrs_normalizer);
        obj.msrs_normalizer_trnsp = obj.toFloat(obj.msrs_normalizer_trnsp);
      end
      obj.nrm = obj.toCPUFloat(obj.nrm);
      obj.nrm_inv = obj.toCPUFloat(obj.nrm_inv);
    end
    
    function ofst_msrs = getEdgeMsrmnts(obj, ofst_msrs, ofsts_list, nghbr_list)
      % getEdgeMsrmnts computes edge detection on measurements
      %   Input arguments
      %     ofst_msrs - An array of measurements (or features), each column
      %                 corresponding to a particular measurement
      %     ofsts_list - an vector of column indices which are of interest.
      %     nghbr_list - an matrix of column indices. The number of rows is the
      %                  length of ofsts_list. Each row contains indices of
      %                  columns which are neighbors of the given column.
      %  Output arguments
      %    ofst_msrs - created by taking a sub-matrix of ofst_msrs, consisting
      %                of the columns listed in ofsts_list, and from each
      %                column subtracting pointwise average of the
      %                corresponding nghbr_list columns.
      prev_msrs = ofst_msrs;
      %     ofst_msrs = builtin('zeros', size(prev_msrs,1), length(ofsts_list));
      ofst_msrs = obj.zeros(size(prev_msrs,1), length(ofsts_list));
      for k=1:length(ofsts_list)
        ofst_msrs(:,k) = prev_msrs(:,ofsts_list(k)) - ...
          (1/size(nghbr_list,2))*sum(prev_msrs(:,nghbr_list(k,:)'),2);
      end
    end
    
  end
  
  methods (Static, Access=protected)
  function y = compMsrsNormalizer(obj)
    y = obj.zeros(obj.nCols(),1);
    for k=1:length(y)
      v = obj.zeros(obj.nCols(),1);
      v(k) = 1;
      y(k) = norm(obj.multTrnspVec(v),2);
    end
  end
  
  end
  
  methods (Static, Access=protected)
    function ign = ignoreInEqual()
      ign = {'matrix', 'nrm', 'nrm_inv', 'exct_nrm'};
    end
  end
end

