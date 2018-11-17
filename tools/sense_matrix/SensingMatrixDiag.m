classdef SensingMatrixDiag < SensingMatrix
  %SensingMatrixDiag - A diagonal matrix
  
  properties
    mltplr = [];
  end
  
  methods
    % Constructor
    %   Input
    %     diag_vec - elements of the diagonal
    function obj = SensingMatrixDiag(varargin)
      obj.setSensingMatrixDiag(varargin{:});
    end
    
    % Initialize the matrix
    %   Input
    %     obj - this object
    %     diag_vec - elements of the diagonal
    function set(varargin)
      varargin = parseInitArgs(varargin, {'diag_mlt'});
      obj.setSensingMatrixDiag(varargin{:});
    end
    
    % Since the matrix is symmetric, transpose does not change
    % anything.
    function transpose(~)
    end
    
    function dg = getDiag(obj)
      dg = obj.toFloat(obj.mltplr(:));
    end
    
    %change the matrix to be its inverse
    function invert(obj)
      obj.mltplr = 1 ./ obj.mltplr;
      t = 1 ./ [obj.nrm, obj.nrm_inv];
      obj.nrm = t(2);
      obj.nrm_inv = t(1);
      obj.exct_nrm = obj.exct_nrm([2 1]);
      obj.matrix = [];
    end
    
    % doMultVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      y = obj.mltplr(:) .* x(:);
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      y = obj.doMultVec(x);
    end
    
    function y = doMultMat(obj,x)
      y = obj.zeros(size(x));
      if size(x,1) >= size(x,2)
        for k=1:size(x,2)
          y(:,k) = obj.mltplr(:) .* x(:,k);
        end
      else
        for k=1:size(x,1)
          y(k,:) = obj.mltplr(k) * x(k,:);
        end
      end
    end
    
    function y = doMultTrnspMat(obj,x)
      y = obj.doMultMat(x);
    end
    
    % Multiply a matrix or a vector by the matrix whose entries are the
    % absolute values of the entries of this matrix.
    function y = multAbs(obj, x)
      d = abs(obj.mltplr(:));
      y = obj.zeros(length(d), size(x,2));
      for k = 1:size(y,2)
        y(:,k) = d .* x;
      end
    end
      
    function mtrx = doCompMatrix(obj)
      if ~obj.use_gpu && ~obj.use_single
        indcs = double(1:obj.nRows())';
        v = double(obj.mltplr);
        mtrx = sparse(indcs, indcs, v, indcs(end), indcs(end));
      else
        dg = obj.getDiag();
        len = length(dg);
        mtrx = obj.zeros(len,len);
        for k=1:len
          mtrx(k,k) = dg(k);
        end          
      end
    end
    
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(obj,y)
      y = y ./ obj.mltplr;
    end
    
    % undo the operation of normalizeMsrs
    function y = deNormalizeMsrs(obj,y)
      y = y .* obj.mltplr;
    end
    
    % returns true if getMatrix returns a sparse matrix
    function is_sprs = isSparse(~)
      is_sprs = true;
    end
    
    function val = isDiagonal(~)
      val = true;
    end
    
    % Set an exact value for norm. It can be computationally heavy
    function val = cmpExactNorm(obj)
      val = obj.toCPUFloat(gather(max(abs(obj.mltplr))));
    end

    function L = do_compSVD1(obj, ~)
      dg = obj.getDiag();
      dg = sort(dg(:),1,'descend');
      if ~complete
        L = SensingMatrixDiag.constructDiag(dg);
      else
        L = SensingMatrixDiag.constructDiag(dg);
      end
    end
    
    function [L,U,V] = do_compSVD3(obj, ~)
      nc = obj.nCols();
      L = obj.copy();
      U = SensingMatrixUnit(nc);
      V = SensingMatrixUnit(nc);
      [L,U,V] = obj.sortSVD(L,U,V);
    end
    
    function mtx = copyScale(obj, scl)
      if scl == 1
        mtx = copy(obj);
      else
        mtx = SensingMatrixDiag(scl * obj.getDiag());
      end
    end
  end
  
  methods(Static)
     
    function mtx = construct(varargin)
      mtx = SensingMatrixDiag.constructDiag(varargin{:});
    end
    
    % Returns a diagonal sensing matrix with diagonal dg
    %   Input:
    %     dg - a vector of the diagonal elements. If empty a assume it is
    %          a unit matrix of size 0x0.
    %     nr - number of rows (optional)
    %     nr - number of columns (optional)
    function mtx = constructDiag(dg, nr, nc)
      if nargin < 3
        if nargin < 2
          nr = length(dg);
        end
        nc = length(dg);
      end
      
      if any([nr,nc] < length(dg))
        error('One of [nr,nc]=[%d,%d] < length(dg) = %d', nr, nc, length(dg));
      elseif any([nr,nc] > length(dg))
        mtrx = cell(3,1);
        n_mtrx = 0;
        if nr > length(dg)
          n_mtrx = n_mtrx+1;
          mtrx{n_mtrx} = SensingMatrixSelectRange(1,length(dg),nr);
          mtrx{n_mtrx}.transpose();
        end
        n_mtrx = n_mtrx+1;
        mtrx{n_mtrx} = SensingMatrixDiag.constructDiag(dg);
        if nc > length(dg)
          n_mtrx = n_mtrx+1;
          mtrx{n_mtrx} = SensingMatrixSelectRange(1,length(dg),nc);
        end
        
        mtx = SensingMatrixCascade.constructCascade(mtrx);
      elseif isempty(dg)
         mtx = SensingMatrixUnit(length(dg));
      elseif all(dg == dg(1))
        if dg(1) == 1
          mtx = SensingMatrixUnit(length(dg));
        else
          mtx = SensingMatrixScaler(length(dg), dg(1));
        end
      else
        mtx = SensingMatrixDiag(dg);
      end
    end
  end
   
  methods (Access=protected)
    % Initialize the matrix
    %   Input
    %     obj - this object
    %     diag_vec - elements of the diagonal
    function setSensingMatrixDiag(obj, diag_mlt)
      if nargin < 2
        sm_args = {};
      else
        sm_args = {length(diag_mlt), length(diag_mlt)};
      end
      obj.setSensingMatrix(sm_args{:});
      obj.setPsdOrtho_col(true);
      obj.setPsdOrtho_row(true);
      if nargin < 2
        obj.mltplr = obj.toFloat([]);
      else
        obj.mltplr = obj.toFloat(diag_mlt(:));
      end
      
      obj.setSparse(true);
    end
    
    function setCastFloat(obj)
      obj.setCastFloat@SensingMatrix();
      obj.mltplr = obj.toFloat(obj.mltplr);
    end
  end
  
end

