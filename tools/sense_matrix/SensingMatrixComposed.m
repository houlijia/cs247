classdef SensingMatrixComposed < SensingMatrix
  %SensingMatrixComposed An abstract class of sensing matrices composed
  %of other matrices.
  %
  
  properties
    mtrx=[];  % Cell array n_mtrx of matrices
  end
  
  methods
    function obj = SensingMatrixComposed(varargin)
      % Constructor can have 0,1 or 2 arguments.
      %   Input
      %     obj - this object
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      %     nrm_aa (optional) norm A'A
      varargin = parseInitArgs(varargin, {'mtrcs', 'nrm_aa'});
      obj.setSensingMatrixComposed(varargin{:})
    end
    
    function set(obj, varargin)
      %   Input
      %     obj - this object
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      %     nrm_aa (optional) norm A'A
      varargin = parseInitArgs(varargin, {'mtrcs','nrm_aa'});
      obj.setSensingMatrixComposed(varargin{:})
    end
    
    % Clear the computed matrix
    function clearMatrix(obj)
      obj.clearMatrix@SensingMatrix();
      for k=1:length(obj.mtrx)
        obj.mtrx{k}.clearMatrix();
      end
    end
    
    function init(obj, mtrcs)
      obj.set(mtrcs);
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
    
  end
  
  methods (Access=protected)
    function otr = copyElement(obj)
      otr = copyElement@SensingMatrix(obj);
      
      for k=1:numel(obj.mtrx)
        otr.mtrx{k} = copy(obj.mtrx{k});
      end
    end
    
    function setSensingMatrixComposed(obj, mtrcs, nrm_aa)
      % Initialize then object
      %   Input
      %     obj - this object
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      %     nrm_aa (optional) norm A'A
      obj.setSensingMatrix();
      
      if nargin > 1
        obj.setMtrcs(mtrcs);
        if nargin > 2
          obj.setNorm(nrm_aa);
        end
      end
      
      % estimate sparsity
      is_sprs = true;
      for k=1:length(obj.mtrx)
        if ~obj.mtrx{k}.isSparse()
          is_sprs = false;
          break;
        end
      end
      obj.setSparse(is_sprs);
    end
    
    function setMtrcs(obj, mtrcs)
      n = length(mtrcs);
      if ~n
        obj.mtrx = [];
        return;
      end
      
      if isstruct(mtrcs)
        mtrcs = arrayfun(@(x) {SensingMatrix.construct(x.type, x.args)}, mtrcs);
      else
        for k=1:numel(mtrcs)
          if ~isa(mtrcs{k}, 'SensingMatrix')
            mtrcs{k} = SensingMatrix.construct(...
              mtrcs{k}.type, mtrcs{k}.args);
          else
            mtrcs{k} = copy(mtrcs{k});
          end
        end
      end
      [ncl, nrw, orth, psd_orth] = obj.compDim(mtrcs);
      
      obj.setSensingMatrix(nrw, ncl);
      if ~isrow(mtrcs)
        mtrcs = mtrcs';
      end
      obj.mtrx = mtrcs;
      obj.setOrthoCol(orth.col);
      obj.setOrthoRow(orth.row);
      obj.setPsdOrthoCol(psd_orth.col);
      obj.setPsdOrthoRow(psd_orth.row);
    end
    
    function setUseGpu(obj,val)
      for k=1:numel(obj.mtrx)
        obj.mtrx{k}.use_gpu = val;
      end
      obj.setUseGpu@SensingMatrix(val);
    end
    
    function setUseSingle(obj,val)
      for k=1:numel(obj.mtrx)
        obj.mtrx{k}.use_single = val;
      end
      obj.setUseSingle@SensingMatrix(val);
    end
  end
  
  methods (Access=protected, Abstract)
    % Compute the matrix dimensions,
    [ncl, nrw, nnclp] = compDim(obj, mtrcs)
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

