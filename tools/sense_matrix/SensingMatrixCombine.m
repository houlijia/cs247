classdef SensingMatrixCombine < SensingMatrixComposed
  %SensingMatrixCombine is a linear combination of several sensing
  %matrices of the same order
  %   Detailed explanation goes here
  
  properties
    wgt=[];   % a vector of n_mtrx weights
  end
  
  methods
    function obj = SensingMatrixCombine(varargin)
      % Constructor can have 0,2 or 3 arguments.
      %   Input
      %     obj - this object
      %     wg - an array of weights of matrices
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      %     nrm_aa (optional) norm A'A
      obj.setSensingMatrixCombine(varargin{:})
    end
    
    function set(obj, varargin)
      %   Input
      %     obj - this object
      %     wg - an array of weights of matrices
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      %     nrm_aa (optional) norm A'A
      varargin = parseInitArgs(varargin, {'wg', 'mtrcs', 'nrm_aa'});
      obj.setSensingMatrixCombine(varargin{:})
    end
    
    function setWgt(obj, wg)
      if iscell(wg)
        obj.wgt = obj.zeros(size(wg));
        for k=1:numel(wg)
          obj.wgt(k) = obj.toFloat(obj.wgt{k});
        end
      else
        obj.wgt = obj.toFloat(wg);
      end
    end
    
    % doMultVec - Multiply a vector x of length nCols() by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      y = obj.zeros(obj.n_rows, 1);
      for k=1:length(obj.wgt)
        y = y + obj.wgt(k)*obj.mtrx{k}.multVec(x);
      end
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length nCols().  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      y = obj.zeros(obj.n_cols, 1);
      for k=1:length(obj.wgt)
        y = y + obj.wgt(k)*obj.mtrx{k}.multTrnspVec(x);
      end
    end
    
    % doMultMat - Multiply a matrix x with n_cols rows by the matrix and return
    % a vector y of length n_rows.
    function y = doMultMat(obj, x)
      y = obj.zeros(obj.n_rows, size(x,2));
      for k=1:length(obj.wgt)
        y = y + obj.wgt(k)*obj.mtrx{k}.multMat(x);
      end
    end
    
    % doMultTrnspMat - Multiply a matrix x with n_rows rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspMat(obj, x)
      y = obj.zeros(obj.n_cols, size(x,2));
      for k=1:length(obj.wgt)
        y = y + obj.wgt(k)*obj.mtrx{k}.multTrnspMat(x);
      end
    end
    
    function mtrx = doCompMatrix(obj)
      if ~obj.is_transposed
        mtrx = obj.wgt(1) * obj.mtrx{1}.getMatrix();
        for k=2:length(obj.mtrx)
          mtrx = mtrx + obj.wgt(k)*obj.mtrx{k}.getMatrix();
        end
      else
        mtx = obj.mtrx{1};
        mtx.transpose();
        mtrx = obj.wgt(1) * mtx;
        mtx.transpose();
        for k=2:length(obj.mtrx)
          mtx = obj.mtrx{k};
          mtx.transpose();
          mtrx = mtrx + obj.wgt(k)*mtx.getMatrix();
          mtx.transpose();
        end
      end
    end
    
  end
  
  methods (Access=protected)
    function setSensingMatrixCombine(obj, wg, mtrcs, nrm_aa)
      %   Input
      %     obj - this object
      %     wg - an array of weights of matrices
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      %     nrm_aa (optional) norm A'A
      switch nargin
        case 1
          sm_args = {};
        case 3
          sm_args = {mtrcs};
        case 4
          sm_args = {mtrcs, nrm_aa};
      end
      obj.setSensingMatrixComposed(sm_args{:});
      
      if nargin > 1
        obj.setWgt(wg);
      end
      
      function indcell = get_indcs(k, trp)
        indcell = obj.mtrx{k}.indcsNoClip(trp);
        indcell = {indcell};
      end
      
      for trnsp = 0:1
        indcs = arrayfun(@(k) get_indcs(k, trnsp), (1:numel(obj.mtrx))');
        indcs = vertcat(indcs{:});
        obj.setIndcsNoClip(unique(indcs), trnsp);
      end
      
    end
    
    function [ncl, nrw, orth, psd_orth] = compDim(~, mtrcs)
      nrw = mtrcs{1}.n_rows;
      ncl = mtrcs{1}.nCols();
      if length(mtrcs) == 1
        orth = struct('col', mtrcs{1}.getOrthoCol(), ...
          'row', mtrcs{1}.getOrthoRow());
        psd_orth = struct('col', mtrcs{1}.getPsdOrthoCol(), ...
          'row', mtrcs{1}.getPsdOrthoRow());
      else
        orth = struct('col', false,  'row',false);
        psd_orth = struct('col', false,  'row',false);
      end
      for k=2:length(mtrcs)
        mtx = mtrcs{k};
        if nrw ~= mtx.n_rows || ncl ~= mtx.nCols()
          error('not all matrices have same dimensions');
        end
      end
    end
    
    % This function compute an upper bound, not the actual norm
    function nrm = compNorm(obj)
      n_mtrx = length(obj.mtrx);
      nrm_aa = arrayfun(@(k) obj.mtrx{k}.norm(), (1:n_mtrx)');
%       nrm_aa = obj.toCPUFloat(zeros(n_mtrx,1));
%       for k=1:n_mtrx
%         nrm_aa(k) = obj.mtrx{k}.norm();
%       end
      nrm = dot(abs(gather(obj.wgt)), nrm_aa);
    end
  
    function setUseGpu(obj,val)
      obj.setUseGpu@SensingMatrixComposed(val);
      obj.wgt = obj.toFloat(obj.wgt);
    end
      end
 end

