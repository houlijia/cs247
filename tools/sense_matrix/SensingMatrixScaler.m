classdef SensingMatrixScaler < SensingMatrixDiag
  %SensingMatrixScaler - A square matrix which mutliplies a vector by a scaler
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    % Constructor
    %   Input
    %     order - order of the square matrix
    %     mlt - multiplier (default = 1)
    function obj = SensingMatrixScaler(varargin)
      obj.setSensingMatrixScaler(varargin{:});
    end
    
    % Initialize the matrix
    %   Input
    %     obj - this object
    %     order - order of the square matrix
    %     mlt - multiplier (default = 1)
    function set(varargin)
      varargin = parseInitArgs(varargin, {'order', 'mlt'});
      obj.setSensingMatrixScaler(varargin{:});
    end
    
    function len=decode(obj, code_src, ~, cnt)
      if nargin < 4
        cnt = inf;
      end
      
      % Decode order
      [vals, n_read] = code_src.readUInt(cnt,[1,1]);
      if ischar(vals) || (isscalar(vals) && vals == -1)
        if ischar(vals)
          len = vals;
        else
          len = 'Unexpected end of data';
        end
        return;
      end
      len = n_read;
      cnt = cnt - n_read;
      ordr = vals(1);
      
      [mlt, n_read] = code_src.readNumber(cnt, 1);
      if ischar(mlt)
        if isempty(mlt)
          mlt = 'Unexpected end of data';
        end
        len = mlt;
        return
      end
      len = len + n_read;
      
      obj.setSensingMatrixScaler(ordr, mlt);
    end
    
    function scl = scaler(obj)
      scl = obj.mltplr;
    end
    
    function dg = getDiag(obj)
      dg = obj.mltplr * obj.ones(obj.nCols(),1);
    end
    
    % doMultVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      y = obj.mltplr * x;
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      y = obj.mltplr * x;
    end
    
    function y = doMultMat(obj,x)
      y = obj.mltplr * x;
    end
    
    function y = doMultTrnspMat(obj,x)
      y = obj.mltplr * x;
    end
    
    % Multiply a matrix or a vector by the matrix whose entries are the
    % absolute values of the entries of this matrix.
    function y = multAbs(obj, x)
      y = abs(obj.mltpr) * x;
    end
    
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(obj,y)
      y = y / obj.mltplr;
    end
    
    % undo the operation of normalizeMsrs
    function y = deNormalizeMsrs(obj,y)
      y = y * obj.mltplr;
    end
    
    % Set an exact value for norm. It can be computationally heavy
    function y = cmpExactNorm(obj)
      y = obj.toCPUFloat(abs(gather(obj.mltplr)));
    end
    
    function mtx = copyScale(obj, scl)
      scl = scl * obj.scaler();
      if scl == 1
        mtx = SensingMatrix;
      else
        mtx = SensingMatrixScaler(obj.nCols(), scl);
      end
    end
    
  end
  
  methods(Static)
    function mtx = construct(scl, nr, nc)
      % Generate a matrix whose diagonal elements are scl
      % Input 
      %   scl - the value in the diagonal. If nargin==1, this is a struct with
      %         the following fields:
      %           order - order of the square matrix
      %           mlt - multiplier (default = 1)
      %           
      %   nr - number of rows
      %   nc - number of columns (optional. Default: nr)
      if nargin == 1 && isstruct(scl)
        args = scl;
        nr = args.order;
        nc = nr;
        scl = args.mlt;
      elseif nargin < 3
        nc = nr;
      end
      
      if scl == 1
        mtx = SensingMatrixUnit(nr);
      else
        mtx = SensingMatrixScaler(nr,scl);
      end
      if nr == nc
        return
      end
      if nr < nc
        mtx = SensingMatrixCascade({mtx, ...
          SensingMatrixSelectRange(1,nr,nc,false)});
      else
        mtx = SensingMatrixCascade({...
          SensingMatrixSelectRange(1,nc,nr,true), mtx});
      end
    end
  end
  
  methods (Access=protected)
    % Initialize the matrix
    %   Input
    %     obj - this object
    %     order - order of the square matrix
    %     mlt - multiplier (default = 1)
    function setSensingMatrixScaler(obj, order, mlt)
      if nargin < 2
        sm_args = {};
      else
        sm_args = {order, order};
      end
      
      obj.setOrtho(true);
      
      obj.setSensingMatrix(sm_args{:});

      if nargin == 1
        return
      end
      
      if nargin < 3
        obj.mltplr = obj.toCPUFloat(1);
      else
        obj.mltplr = obj.toCPUFloat(mlt);
      end
    end
    
    function mtx = create(obj, args)
      mtx = obj.construct(args);
    end
    
    function setCastFloat(obj)
      obj.setCastFloat@SensingMatrix();
      obj.mltplr = obj.toCPUFloat(obj.mltplr);
    end
    
  end
  
end

