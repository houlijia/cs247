classdef SensingMatrixSelect < SensingMatrix
  %SensingMatrixSelect is a matrix where each row is all zeros except for
  %one entry of 1.  Essentially multiplying by the matrix retruns a
  %a selection of the entries of the original vector.
  
  properties
    slct_indices=[];
  end
  
  methods
    % constructor can be called with either 0, 2 or 3 arguments.
    % Input:
    %   indcs - an array of indices selected by the matrix (its length
    %           is the number of rows)
    %   num_columns - number of columns
    %     transpose: (optional) If present and true, the matrix is transposed.
    function obj = SensingMatrixSelect(varargin)
      obj.setSensingMatrixSelect(varargin{:})
    end
    
    % Initialize the matrix
    % Input:
    %   obj - this object
    %   indcs - an array of indices selected by the matrix (its length
    %           is the number of rows)
    %   num_columns - number of columns
    %     transpose: (optional) If present and true, the matrix is transposed.
    function set(obj, varargin)
      varargin = parseInitArgs(varargin, {'indcs', 'num_columns', 'transpose'});
      obj.setSensingMatrixSelect(varargin{:})
    end
    
    function indcs = getIndices(obj)
      indcs = obj.slct_indices;
    end
    
    % Returns true if the matrix does a permutation on the vector, that is,
    % if it is square and all indices are different
    function val = isPermutation(obj)
      nr = obj.nRows();
      val = (nr == obj.nCols() && nr == length(unique(obj.getIndices)));
    end
       
    function dg = getDiag(obj)
      dg = double(obj.slct_indices(:) == (1:obj.n_rows)');
    end
    
    function val = isDiagonal(obj)
      val = all(obj.getDiag());
    end
    
   % doMultVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      y = x(obj.slct_indices);
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return a vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      y = obj.zeros(obj.n_cols, 1);
      y(obj.slct_indices) = x;
    end
    
    function y = doMultMat(obj, x)
      y = x(obj.slct_indices,:);
    end
    
    function y = doMultTrnspMat(obj, x)
      y = obj.zeros(obj.n_cols, size(x,2));
      y(obj.slct_indices,:) = x;
    end
        
    % Redefine getMtrx as getting a sparse matrix
    function mtrx = doCompMatrix(obj)
      if obj.use_gpu || obj.use_single
        mtrx = obj.zeros(obj.nRows(), obj.nCols());
        if ~obj.is_transposed
          for k=1:length(obj.slct_indices)
            mtrx(k,obj.slct_indices(k)) = 1;
          end
        else
          for k=1:length(obj.slct_indices)
            mtrx(obj.slct_indices(k),k) = 1;
          end
        end
      else
        si = double(gobj.slct_indices);
        rw = (1:length(si));
        nr = double(obj.n_rows);
        nc = double(obj.n_cols);
        if ~obj.is_transposed
          mtrx = sparse(rw, si, 1, nr, nc);
        else
          mtrx = sparse(si, rw, 1, nc, nr);
        end
      end
    end
    
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(~,y)
    end
    
    % undo the operation of normalizeMsrs
    function y = deNormalizeMsrs(~,y)
    end
    
    % returns true if getMatrix returns a sparse matrix
    function is_sprs = isSparse(~)
      is_sprs = true;
    end
    
    % Set an exact value for norm. It can be computationally heavy
    function val = cmpExactNorm(~)
      val = 1;
    end
    
    function L = do_compSVD1(obj, complete)
      if obj.is_transposed
        obj.transpose();
        L = obj.do_compSVD1(complete);
        obj.transpose();
        L = obj.transposeSVD(L);
        return
      end
      
      if complete
        L = obj.construct((1:obj.nRows())', obj.nCols(), false);
      else
        L = SensingMatrixUnit(obj.nRows());
      end
    end
    
    function [L,U,V] = do_compSVD3(obj,complete)
      if obj.is_transposed
        obj.transpose();
        [L,U,V] = obj.do_compSVD3(complete);
        obj.transpose();
        [L,U,V] = obj.transposeSVD(L,U,V);
        return
      end
      
      if complete
        cmplmnt = (1:obj.nCols())';
        indcs = obj.getIndices();
        cmplmnt(indcs(:)) = [];
        V = obj.construct([indcs;cmplmnt], obj.nCols(), true);
        L = obj.construct((1:obj.nRows())', obj.nCols(), false);
      else
        V = obj.copy();
        V.transpose();
        L = SensingMatrixUnit(obj.nRows());
      end
      U = SensingMatrixUnit(obj.nRows());
    end
  end
  
  methods (Static)
    % Create a matrix of class SensingMatrixSelect (or a sub class or a unit matrix).
    %   indcs - an array of indices selected by the matrix (its length
    %           is the number of rows). If there is only one argument, than it
    %           is a struct with the following fields:
    %     indcs - an array of indices selected by the matrix (its length
    %           is the number of rows)
    %     num_columns - number of columns
    %     transpose: (optional) If present and true, the matrix is transposed.
    %   num_columns - number of columns
    %   trnsp - Optional: If true, make the matrix transposed
    % Output:
    %   mtx - output matrix
    function mtx = construct(indcs, num_columns, trnsp)
      if nargin <= 1 && isstruct(indcs)
        args = indcs;
        if isfield(args, 'trnsp')
          trnsp = args.trnsp;
        else
          trnsp = false;
        end
        num_columns = args.num_columns;
        indcs = args.indcs;
      elseif nargin <= 2
        trnsp = false;
      end
      
      rbgn = min(indcs(:));
      rend = max(indcs(:));
      if isequal(indcs(:), (rbgn:rend)')
        if length(indcs) == num_columns
          mtx = SensingMatrixUnit(num_columns);
        else
          mtx = SensingMatrixSelectRange(rbgn,rend,num_columns);
        end
      else
        mtx = SensingMatrixSelect(indcs, num_columns);
      end
      
      if trnsp
        mtx.transpose();
      end
    end
  end
  
  methods (Access=protected)
    % Initialize the matrix
    % Input:
    %   obj - this object
    %   indcs - an array of indices selected by the matrix (its length
    %           is the number of rows)
    %   num_columns - number of columns
    %   transpose: (optional) If present and true, the matrix is transposed.
    function setSensingMatrixSelect(obj, indcs, num_columns, transpose)
      if nargin <= 1
        sm_args  = {};
      else
        sm_args = {length(indcs), num_columns};
      end
      
      obj.setSensingMatrix(sm_args{:});
      obj.setSparse(true);
      
      if nargin > 1
        obj.slct_indices = obj.toIndex(indcs(:));
        if length(indcs) == length(unique(indcs(:)))
          obj.setOrtho_row(true);
          if obj.n_rows == obj.n_cols
            obj.setOrtho_col(true);
          else
            obj.setPsdOrtho_col(true);
          end
        else
          obj.setPsdOrtho_col(true);
        end
      
        if nargin >=4 && transpose
          obj.transpose();
        end
      end
    end
    
    function mtx = create(obj, args)
      mtx = obj.construct(args);
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrix();
      obj.slct_indices = obj.toIndex(obj.slct_indices);
    end
  end
end
  
