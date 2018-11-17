classdef SensingMatrixBasicDCT < SensingMatrix
  methods
    % Constructor
    %   Input:
    %     n_cols - number of columns
    function obj = SensingMatrixBasicDCT(varargin)
      obj.set(varargin{:});
    end
    
    function set(obj, varargin)
      varargin = parseInitArgs(varargin, {'n_cols'});
      obj.setSensingMatrixBasicDCT(varargin{:});
    end
    
    % Returns the sum of values of the measurement which contain DC value,
    % weighted by the ratio of the DC value to other components (in
    % terms of RMS), or 0 if there is no such measurement.
    %   Input:
    %     obj - this object
    %     msrs - the measurements vector
    function dc_val = getDC(~, msrs)
      dc_val = msrs(1);  %default when DC is not known
    end
    
    
    function y = doMultVec(obj, x)
       % doMultVec - DCT transform a vector x of length nCols()
      y = dct(x, obj.nCols());
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
       % doMultVec - Inverse DCT transform a vector x of length nCols()
      y = idct(x, obj.nRows());
    end
    
    function y = doMultMat(obj, x)
       % doMultVec - DCT transform each column of a matrix x with nCols()
       % rows.
      y = dct(x, obj.nCols());
    end
    
    function y = doMultTrnspMat(obj, x)
       % doMultVec - DCT transform each column of a matrix x with nCols()
       % rows.
      y = idct(x, obj.nCols());
    end
    
    % Set an exact value for Norm. It can be computationally heavy
    function val = cmpExactNorm(~)
      val = 1;
    end
    
    function L = do_compSVD1(obj, ~)
      L = SensingMatrixUnit(obj.nCols());
    end
    
    function [L,U,V] = do_compSVD3(obj, ~)
      L = SensingMatrixUnit(obj.nCols());
      U = obj.copy();
      V = L.copy();
    end
  end
    
  methods (Access=protected)
    function setSensingMatrixBasicDCT(obj, n_cols)
      if nargin >= 2
        obj.setSensingMatrix(n_cols, n_cols);
      end
      obj.setOrtho(true);
      obj.setIndcsNoClip(1);
    end
    
    function y = compNorm(~)
      y = 1;
    end
    
  end
end
