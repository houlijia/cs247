classdef SensingMatrixMatlab < SensingMatrix
  %SensingMatrixMatlab is an encapsulation of a Matlab matrix
  %   Detailed explanation goes here
  
  methods
    function obj=SensingMatrixMatlab(varargin)
      varargin = parseInitArgs(varargin, {'mtrx'});
      obj.setSensingMatrixMatlab(varargin{:});
    end
    
    function set(obj, varargin)
      obj.setSensingMatrixMatlab(varargin{:});
    end
    
    %change the matrix to be its inverse
    function invert(obj)
      obj.mtrx = inv(obj.mtrx);
      obj.matrix = [];
      t = 1 ./ [obj.nrm, obj.nrm_inv];
      obj.nrm = t(2);
      obj.nrm_inv = t(1);
      obj.exct_nrm = obj.exct_nrm([2 1]);
    end
    
    % returns true if getMatrix returns a sparse matrix
    function is_sprs = isSparse(obj)
      is_sprs = issparse(obj.mtrx);
    end
    
    function y = doMultVec(obj, x)
      y = obj.mtrx * x;
    end
    
    function y = doMultTrnspVec(obj, x)
      y = obj.mtrx' * x;
    end
    
    function y = doMultMat(obj, x)
      y = obj.mtrx * x;
    end
    
    function y = doMultTrnspMat(obj, x)
      y = obj.mtrx' * x;
    end
    
    % Computes the matlab matrix which corresponds to what this matrix
    % should be.
    function mtx = doCompMatrix(obj)
      if obj.is_transposed
        mtx = obj.mtrx';
      else
        mtx = obj.mtrx;
      end
    end
    
    function dg = getDiag(obj)
      dg = diag(obj.mtrx);
    end
    
    function val = isDiagonal(obj)
      if isempty(obj.is_diag)
        if isvector(obj.mtrx)
          obj.is_diag = all(obj.mtrx(2:end)==0);
        else
          dg = diag(obj.mtrx);
          ndg = length(dg);
          mtx = obj.mtrx;
          mtx(1:ndg,1:ndg) = mtx(1:ndg,1:ndg) - diag(dg);
          obj.is_diag = ~any(mtx(:));
        end
      end
      val = obj.is_diag;
    end
  end
  
  methods (Access=protected)
    function setSensingMatrixMatlab(obj, mtx)
      if nargin >1
        sm_args = {size(mtx,1), size(mtx,2)};
      else
        sm_args = {};
      end
      
      obj.setSensingMatrix(sm_args{:});
      
      if nargin > 1
        obj.mtrx = obj.toFloat(mtx);
        obj.setSparse(issparse(obj.mtrx));
      end
    end
    
    function y = compMsrsNormalizer(obj)
      mtx = obj.mtrx;
      if ~obj.is_transposed
        mtx = mtx';
      end
      y = sqrt(sum(mtx .* mtx))';
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrix();
      obj.is_diag = obj.toLogical(obj.is_diag);
    end

    function setCastFloat(obj)
      obj.setCastFloat@SensingMatrix();
      obj.mtrx = obj.toFloat(obj.mtrx);
      obj.setSparse(issparse(obj.mtrx));
    end
    
  end
  
  properties (Access=private)
    mtrx = [];
    is_diag = [];
  end
  
end

