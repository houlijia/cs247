classdef SensingMatrixSelectRange < SensingMatrixSelect
  %SensingMatrixSelectRange has the same function as SensingMatrixSelect,
  %but the slct_indices is a range rng_bgn:rng_end.
  
  properties
    rng_bgn;
    rng_end;
  end
  
  methods
    % constructor can be called with either 0, or arguments.
    % Input:
    %   r_bgn - beginning index of range
    %   r_end - ending index of range
    %   num_columns - number of columns
    % There are three cases:
    %   r_end>=r_bgn: Select r_bgn,...,r_end
    %   0<r_end<r_bgn: Select r_bgn,...,num_columns,1,...r_end
    %   r_end==0: Select r_bgn,...,num_columns,1,...,r_bgn-1,
    %             which is equivalent to r_end==r_bgn-1
    function obj = SensingMatrixSelectRange(varargin)
      obj.set(varargin{:})
    end
    
    % Initialize the matrix
    % Input:
    %   obj - this object
    %   r_bgn - beginning index of range
    %   r_end - ending index of range
    %   num_columns - number of columns
    % There are three cases:
    %   r_end>=r_bgn: Select r_bgn,...,r_end
    %   0<r_end<r_bgn: Select r_bgn,...,num_columns,1,...r_end
    %   r_end==0: Select r_bgn,...,num_columns,1,...,r_bgn-1,
    %             which is equivalent to r_end==r_bgn-1
    function set(obj, varargin)
      varargin = parseInitArgs(varargin, {'r_bgn', 'r_end',...
        'num_columns'});
      obj.setSensingMatrixSelectRange(varargin{:})
    end
    
    function indcs = getIndices(obj)
      if obj.rng_bgn <= obj.rng_end
        indcs = (obj.rng_bgn:obj.rng_end)';
      else
        indcs = [(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)]';
      end
    end
    
    function dg = getDiag(obj)
      if  obj.rng_bgn == 1
        dg = obj.ones(obj.rng_end,1);
      else
        dg = obj.zeros(obj.rng_end,1);
      end
    end

    function val = isDiagonal(obj)
      val = (obj.rng_bgn == 1);
    end

    % doMultVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      if obj.rng_bgn <= obj.rng_end
        y = x(obj.rng_bgn:obj.rng_end);
      else
        y = x([(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)]);
      end
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return a vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      y = obj.zeros(obj.n_cols, 1);
      
      if obj.rng_bgn <= obj.rng_end
        y(obj.rng_bgn:obj.rng_end) = x;
      else
        y([(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)]) = x;
      end
    end
    
    function y = doMultMat(obj, x)
      if obj.rng_bgn <= obj.rng_end
        y = x(obj.rng_bgn:obj.rng_end,:);
      else
        y = x([(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)],:);
      end
    end
    
    function y = doMultTrnspMat(obj, x)
      y = obj.zeros(obj.n_cols, size(x,2));
      if obj.rng_bgn <= obj.rng_end
        y(obj.rng_bgn:obj.rng_end,:) = x;
      else
        y([(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)],:) = x;
      end
    end
    
    function mtrx = doCompMatrix(obj)
      % Create a sparse array
      nr = obj.n_rows;
      nc = obj.n_cols;
      if ~obj.is_transposed
        if obj.rng_bgn <= obj.rng_end
          rg = double(obj.rng_bgn:obj.rng_end);
        else
          rg = double([(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)]);
        end
        if ~obj.use_gpu && ~obj.use_single
          mtrx = sparse(double(1:nr), double(rg), 1, double(nr), double(nc));
        else
          mtrx = obj.zeros(nr,nc);
          mtrx(sub2ind(size(mtrx), 1:nr, rg)) = 1;
        end
      else
        if obj.rng_bgn <= obj.rng_end
          rg = obj.rng_bgn:obj.rng_end;
        else
          rg = [(obj.rng_bgn:obj.n_cols) (1:obj.rng_end)];
        end
        if ~obj.use_gpu && ~obj.use_single
          mtrx = sparse(double(rg), double(1:nr), 1, double(nc), double(nr));
        else
          mtrx = obj.zeros(nc,nr);
          mtrx(sub2ind(size(mtrx), rg, 1:nr)) = 1;
        end
      end
    end
    
  end
  
  methods (Access=protected)
    % Initialize the matrix
    % Input:
    %   obj - this object
    %   r_bgn - beginning index of range
    %   r_end - ending index of range
    %   num_columns - number of columns
    function setSensingMatrixSelectRange(obj, r_bgn, r_end, ...
        num_columns)
      
      if nargin >= 4
        if r_end == 0
          if r_bgn == 1
            r_end = num_columns;
          else
            r_end = r_bgn-1;
          end
        end
        if r_bgn <= r_end
          obj.setSensingMatrix(r_end-r_bgn+1, num_columns);
        else
          obj.setSensingMatrix(r_end-r_bgn+1+num_columns, num_columns);
        end
        obj.rng_bgn = obj.toCPUIndex(r_bgn);
        obj.rng_end = obj.toCPUIndex(r_end);
      
        obj.setOrtho_row(true);
        if obj.n_rows == obj.n_cols
          obj.setOrtho_col(true);
        else
          obj.setPsdOrtho_col(true);
        end
        obj.setSparse(true);
      end
      
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSelect();
      obj.rng_bgn = obj.toCPUIndex(obj.rng_bgn);
      obj.rng_end = obj.toCPUIndex(obj.rng_end);
    end
  end
  
  methods(Static)
    function mtx = constructSelectRange(r_bgn, r_end, num_columns, trnsp)
      if r_bgn == 1 && r_end == num_columns
        mtx = SensingMatrixUnit(num_columns);
      elseif r_bgn > num_columns || r_bgn <= 0
        mtx = SensingMatrixSelect([],num_columns);
      else
        mtx = SensingMatrixSelectRange(r_bgn, r_end, num_columns);
      end
      if nargin>= 4 && trnsp
        mtx.transpose();
      end
    end
    
    function mtx = construct(varargin)
      mtx = SensingMatrixSelectRange.constructSelectRange(varargin{:});
    end
  end
end


