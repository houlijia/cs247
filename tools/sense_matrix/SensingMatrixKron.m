classdef SensingMatrixKron < SensingMatrixComposed
  % SensingMatrixKron is a Kronecker product of several matrices:
  % mtrx{1} .*. mtrx{2} .*. ... where .*. denotes Kronecker product
  properties (Access=private)
    % dimension of matrics as [num(mtrcs),2] array. For each component matrix,
    % dims[k,1] is the number of rows and dims[k,2] is the number of columns
    dims; 
  end
  
   methods
    function obj = SensingMatrixKron(varargin)
      % Constructor
      %   Input:
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      obj.setSensingMatrixKron(varargin{:})
    end
    
    function set(obj, varargin)
      % Initialize the object
      %   Input:
      %     obj - this object
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      varargin = parseInitArgs(varargin, {'mtrcs'});
      obj.setSensingMatrixKron(varargin{:})
    end
    
    % doMultVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      y = x(:);
      for k=length(obj.mtrx):-1:1
        mtx = obj.mtrx{k};
        nn = obj.dims(k,2);
        y = reshape(y, nn, numel(y)/nn);
        y = mtx.multMat(y);
        y = y';
      end
      y = y(:);
    end
    
    % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      y = x(:);
      for k=length(obj.mtrx):-1:1
        mtx = obj.mtrx{k};
        nn = obj.dims(k,1);
        y = reshape(y, nn, numel(y)/nn);
        y = mtx.multTrnspMat(y);
         y = y';
      end
      y = y(:);
    end
    
    function y = doMultMat(obj, x)
      nc = size(x,2);
      y = x(:);
      for k=length(obj.mtrx):-1:1
        mtx = obj.mtrx{k};
        nn = obj.dims(k,2);
        y = reshape(y, nn, numel(y)/nn);
        y = mtx.multMat(y);
        y = y';
      end
      y = reshape(y, nc, numel(y)/nc);
      y = y';
    end
    
    function y = doMultTrnspMat(obj, x)
      nc = size(x,2);
      y = x(:);
      for k=length(obj.mtrx):-1:1
        mtx = obj.mtrx{k};
        nn = obj.dims(k,1);
        y = reshape(y, nn, numel(y)/nn);
        y = mtx.multTrnspMat(y);
        y = y';
      end
      y = reshape(y, nc, numel(y)/nc);
      y = y';
    end
    
    % Multiply a matrix or a vector by the matrix whose entries are the
    % absolute values of the entries of this matrix.
    function y = multAbs(obj, x)
      nc = size(x,2);
      y = x(:);
      for k=length(obj.mtrx):-1:1
        mtx = obj.mtrx{k};
        nn = obj.dims(k,2);
        y = reshape(y, nn, numel(y)/nn);
        y = mtx.multAbs(y);
        y = y';
      end
      y = reshape(y, nc, numel(y)/nc);
      y = y';
    end
    
    function mtrx = doCompMatrix(obj)
      mtx = obj.mtrx;
      if obj.is_transposed
        for k=1:length(mtx)
          mtx{k}.transpose();
        end
      end
      mtrx = mtx{1}.getMatrix();
      for k=2:length(obj.mtrx)
        mtx1 = mtx{k}.getMatrix();
        if issparse(mtx1) || issparse(mtrx)
          mtrx = sparse(double(gather(kron(mtrx, mtx1))));
        else
          mtrx = kron(mtrx, mtx1);
        end
      end
      if obj.is_transposed
        for k=1:length(mtx)
          mtx{k}.transpose();
        end
      end
    end
    
    function dg = getDiag(obj)
      if obj.is_transposed
        lst = 1:length(obj.mtrx);
      else
        lst = length(obj.mtrx):-1:1;
      end
      
      dg = obj.mtrx{lst(1)}.getDiag();
      for k=lst(2:end)
        dg = dg * (obj.mtrx{k}.getDiag())';
        dg = dg(:);
      end
    end
    
    % Set an exact value for Norm. It can be computationally heavy
    function val = cmpExactNorm(obj)
      nrm_aa = obj.zeros(size(obj.mtrx));
      for k=1:numel(obj.mtrx)
        nrm_aa(k) = obj.mtrx{k}.getExactNorm();
      end
      val = gather(prod(nrm_aa));
    end
    
    %change the matrix to be its inverse
    function invert(obj)
      for k=1:numel(obj.mtrx)
        obj.mtrx{k}.invert();
      end
      t = 1 ./ [obj.nrm, obj.nrm_inv];
      obj.nrm = t(2);
      obj.nrm_inv = t(1);
      obj.exct_nrm = obj.exct_nrm([2 1]);
      obj.matrix = [];
    end
    
    function L = do_compSVD1(obj, complete)
      if obj.is_transposed
        obj.transpose();
        L = obj.do_compSVD1(complete);
        obj.transpose();
        L = obj.transposeSVD(L);
        return
      end
      
      nm = numel(obj.mtrx);
      L = cell(nm,1);
      for k=1:nm
        L{k} = obj.mtrx{k}.do_compSVD1(complete);
      end
      L = obj.compSVDfromTerms(L);
      %           for k=1:nm
      %             L{k} = obj.mtrx{k}.do_compSVD1(complete);
      %             L{k} = L{k}.getDiag();
      %             L{k} = L{k}(:);
      %           end
      %           dg = vertcat(L{:});
      %           dg = sort(dg,1,'descend');
      %           L = SensingMatrixDiag.constructDiag(dg);
    end
    
    function [L,U,V] = do_compSVD3(obj, complete)
      if obj.is_transposed
        obj.transpose();
        [L,U,V] = obj.do_compSVD3(complete);
        obj.transpose();
        [L,U,V] = obj.transposeSVD(L,U,V);
        return
      end
      
      nm = numel(obj.mtrx);
      U = cell(nm,1);
      L = cell(nm,1);
      V = cell(nm,1);
      for k=1:nm
        [L{k},U{k},V{k}] = obj.mtrx{k}.do_compSVD3(complete);
      end
      [L,U,V] = obj.compSVDfromTerms(L,U,V);
      %           if complete
      %             % Sparse representation of L
      %             vv=1;
      %             rr=1;
      %             cc=1;
      %           end
      %           for k=1:nm
      %             [L{k},U{k},V{k}] = obj.mtrx{k}.do_compSVD3(complete);
      %             if complete
      %               % Create a sparse representation of the diagonal matrix,
      %               % taking into account that it is not necessarily square
      %               dg = L{k}.getDiag();
      %               ldg = length(dg);
      %               vv = dg(:) * vv(:)';
      %               rr = [ones(ldg,1); zeros(obj.mtrx{k}.nRows()-ldg,1)] * rr(:)';
      %               cc = [ones(ldg,1); zeros(obj.mtrx{k}.nCols()-ldg,1)] * cc(:)';
      %             end
      %           end
      %           U = obj.constructKron(U);
      %           V = obj.constructKron(V);
      %           if complete
      %             onr = obj.nRows();
      %             onc = obj.nCols();
      %             L = SensingMatrixDiag.constructDiag(vv(:), onr, onc);
      %             P = SensingMatrixSelect.construct([find(rr(:));find(~rr(:))],...
      %               obj.nRows(), true);
      %             U = SensingMatrixCascade.constructCascade({U,P});
      %             P = SensingMatrixSelect.construct([find(cc(:));find(~cc(:))],...
      %               obj.nCols(), true);
      %             V = SensingMatrixCascade.constructCascade({V,P});
      %           else
      %             L = obj.constructKron(L);
      %           end
      %           [L,U,V] = SensingMatrix.sortSVD(L,U,V, 1E-12);
    end
    
    function val = isDiagonal(obj)
      val = true;
      for k=1:numel(obj.mtrx)
        if ~obj.mtrx{k}.isDiagonal()
          val = false;
          break
        end
      end
    end
   end
  
  methods (Static)
    function mtrx = construct(mtrcs)
      mtrx = SensingMatrixKron.constructKron(mtrcs);
    end
    
    function mtrx = constructKron(mtrcs)
      
      function elmnts = get_elmnts(mt)
        if iscell(mt)
          elmnts = cell(1,length(mt));
          for i=1:length(mt)
            elmnts{i} = get_elmnts(mt{i});
          end
          elmnts = horzcat(elmnts{:});
          return
        elseif ~isa(mt,'SensingMatrixKron')
          elmnts = {mt.copy()};
          return
        end
        elmnts = cell(1,length(mt.mtrx));
        if ~mt.is_transposed
          for i=1:length(elmnts)
            elmnts{i} = mt.mtrx{i}.copy();
          end
        else
          for i=1:length(elmnts)
            elmnts{i} = mt.mtrx{i}.copy();
            elmnts{si}.transpose();
            elmnts{i} = get_elmnts(elmnt);
          end
        end
        elmnts = get_elmnts(elmnts);
      end
      
      mtxs = get_elmnts(mtrcs);
      n_mtxs = length(mtxs);
      mtrx = cell(size(mtxs));
      % Discard unity matrix which are scalars
      k=1;
      while k<n_mtxs
        if mtxs{k}.nCols()~=1 || mtxs{k}.nRows() ~=1 || ...
            mtxs{k}.getMatrix() ~= 1
          break;
        end
        k=k+1;
      end
      if k > 1
        mtxs = mtxs(k:end);
        n_mtxs = length(mtxs);
      end
      
      n_mtrx = 1;
      mtrx{n_mtrx} = mtxs{1};
      
      for k=2:n_mtxs
        % Skip unity matrix which are scalars
        if mtxs{k}.nCols()==1 && mtxs{k}.nRows() ==1 && ...
            mtxs{k}.getMatrix() == 1
          continue;
        end
        
        % Combine scaler matrices
        if isa(mtrx{n_mtrx}, 'SensingMatrixScaler') &&...
            isa(mtxs{k}, 'SensingMatrixScaler')
          scl = mtrx{n_mtrx}.scaler() * mtxs{k}.scaler();
          ncl = mtrx{n_mtrx}.nCols() * mtxs{k}.nCols();
          if scl == 1
            mtrx{n_mtrx} = SensingMatrixUnit(ncl);
          else
            mtrx{n_mtrx} = SensingMatrixScaler(ncl, scl);
          end
        else
          n_mtrx = n_mtrx+1;
          mtrx{n_mtrx} = mtxs{k};
        end
      end
      if n_mtrx > 1
        mtrx = SensingMatrixKron(mtrx(1:n_mtrx));
      else
        mtrx = mtrx{1};
      end
    end
    
    % Compute SVD of a Kronecker product matrix from the SVDs of its
    % nT terms.
    %   Input arguments:
    %     LT - A cell array of nT of the diagonal matrices of the SVDs.
    %     UT - A cell array of nT left matrices of the SVDs.
    %     VT - A cell array of nT right matrices of the SVDs.
    %     eps - threshold for singular value differences (default=0). If
    %           specified, chkSVD is computed at the end.
    %     Note that if UT, VT are not specified, U, V are not computed.
    %  Output argument
    %    components of the SVD, such that this matrix is U*L*V', U,V have
    %    orthonormal columns and L is diagonal
    %
    function [L,U,V] = compSVDfromTerms(LT,UT,VT,eps)
      mtx = LT{1};
      toFloat = @mtx.toFloat;
%       toIndex = @mtx.toIndex;
      mk_zeros = @mtx.zeros;
      mk_ones = @mtx.ones;
      if nargin < 4
        eps = 0;
        do_chkSVD = false;
        if nargin < 3 || nargout < 3
          VT = [];
          UT = [];
          U = [];
          V = [];
          if nargin < 2 || nargout < 2
            UT = [];
          end
        end
      else
        do_chkSVD = true;
      end
      
      nT = numel(LT);
      
      % check complete
      if isempty(UT)
        complete = false;
        for k=1:nT
          if LT{k}.nRows() ~= LT{k}.nCols()
            complete = true;
            break
          end
        end
      else
        complete = true;
        for k=1:nT
          if UT{k}.nRows() ~= UT{k}.nCols() || VT{k}.nRows() ~= VT{k}.nCols()
            complete = false;
            break
          end
        end
      end
      
      if ~complete
        L = SensingMatrixKron.constructKron(LT);
      else
        % Sparse representation of L
        vv=toFloat(1);
        rr=toFloat(1);
        cc=toFloat(1);
        onr = toFloat(1);
        onc = toFloat(1);
        for k=1:nT
          dg = LT{k}.getDiag();
          ldg = toFloat(length(dg));
          vv = dg(:) * vv(:)';
          nr = toFloat(LT{k}.nRows());
          nc = toFloat(LT{k}.nCols());
          rr = [mk_ones(ldg,1); mk_zeros(nr-ldg,1)] * rr(:)';
          cc = [mk_ones(ldg,1); mk_zeros(nc-ldg,1)] * cc(:)';
          onr = onr * nr;
          onc = onc * nc;
        end
        L = SensingMatrixDiag.constructDiag(vv(:), onr, onc);
      end
      
      if ~isempty(UT)
        U = SensingMatrixKron.constructKron(UT);
        V = SensingMatrixKron.constructKron(VT);
        if complete
          P = SensingMatrixSelect.construct([find(rr(:));find(~rr(:))],...
            onr, true);
          U = SensingMatrixCascade.constructCascade({U,P});
          P = SensingMatrixSelect.construct([find(cc(:));find(~cc(:))],...
            onc, true);
          V = SensingMatrixCascade.constructCascade({V,P});
        end
        [L,U,V] = SensingMatrix.sortSVD(L,U,V,eps);
        if ~complete
          [L,U,V] = SensingMatrix.truncateSVD(L, U, V);
        end
        if do_chkSVD
          MT = cell(nT,1);
          for k=1:nT
            MT{k} = SensingMatrix.constructSVD(LT{k},UT{k},VT{k});
          end
          M = SensingMatrixKron.constructKron(MT);
          SensingMatrix.chkSVD(L, U, V, 2*eps, M, complete);
        end
      else
        L = SensingMatrix.sortSVD(L, [], [], eps);
        if ~complete
          L = SensingMatrix.truncateSVD(L,[],[]);
        end
        if do_chkSVD
          chkSVD(L, [], [], 2*eps)
        end
      end
    end
  end
  
  methods (Access=protected)
    function setSensingMatrixKron(obj, mtrcs)
      % Initialize the object
      %   Input:
      %     obj - this object
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      switch nargin
        case 1
          sm_args = {};
        case 2
          % take only non-empty matrices
          ne_mtrcs = cell(size(mtrcs));
          cnt = 0;
          for k=1:length(mtrcs)
            if ~isempty(mtrcs{k})
              cnt = cnt+1;
              ne_mtrcs{cnt} = mtrcs{k};
            end
          end
          sm_args = {ne_mtrcs(1:cnt)};
      end
      obj.setSensingMatrixComposed(sm_args{:});
      
      for k=1:length(obj.mtrx)
        if obj.mtrx{k}.isSparse()
          obj.setSparse(true);
          break
        end
      end
      
      obj.setIndcsNoClip(obj.compIndcsNoClip(obj.mtrx, false), false);
      obj.setIndcsNoClip(obj.compIndcsNoClip(obj.mtrx, true),  true);
    end
    
    function [ncl, nrw, orth, psd_orth] = compDim(obj, mtrcs)
      obj.dims = obj.toCPUIndex(zeros(length(mtrcs),2));
      orth = struct('col', obj.toCPULogical(mtrcs{1}.getOrthoCol()), ...
        'row',  obj.toCPULogical(mtrcs{1}.getOrthoRow()));
      psd_orth = struct('col', obj.toCPULogical(mtrcs{1}.getPsdOrthoCol()),...
        'row',  obj.toCPULogical(mtrcs{1}.getPsdOrthoRow()));
      for k=1:length(mtrcs)
        obj.dims(k,1) = obj.toCPUIndex(mtrcs{k}.nRows());
        obj.dims(k,2) = obj.toCPUIndex(mtrcs{k}.nCols());
        orth.col = orth.col && obj.toCPULogical(mtrcs{k}.getOrthoCol());
        orth.row = orth.row && obj.toCPULogical(mtrcs{k}.getOrthoRow());
        psd_orth.col = psd_orth.col && obj.toCPULogical(mtrcs{k}.getPsdOrthoCol());
        psd_orth.row = psd_orth.row && obj.toCPULogical(mtrcs{k}.getPsdOrthoRow());
      end
      rwcl = prod(obj.dims);
      nrw = rwcl(1);
      ncl = rwcl(2);
    end
    
    % This function compute an upper bound, not the actual norm
    function nrm = compNorm(obj)
%       nrm_aa = obj.toCPUFloat(zeros(size(obj.mtrx)));
%       for k=1:numel(obj.mtrx)
%         nrm_aa(k) = obj.mtrx{k}.norm();
%       end
      nrm_aa = arrayfun(@(k) obj.mtrx{k}.norm(), (1:numel(obj.mtrx))');
      nrm = prod(nrm_aa);
    end
    
    function y = doMult(obj, x, dims)
      y = x;
      sz = dims(:,2)';
      for k=1:length(obj.mtrx);
        mtx = obj.mtrx{end-k+1};
        if isa(mtx, 'SensingMatrixScaler')
          % Since we are multiplying by a scalar there is no need for
          % permutations
          
          if ~isa(mtx, 'SensingMatrixUnit')
            y = y * mtx.scaler();
          end
        else
          if k > 1
            y = reshape(y(:),sz);
            prmt = 1:length(sz);
            prmt([1,k]) = prmt([k,1]); % Swap indices 1,k
            y = permute(y,prmt);
            sz_prmt = size(y);
            y = reshape(y(:), [sz(k), numel(y)/sz(k)]);
            y = mtx.multMat(y);
            sz_prmt(1) = dims(k,1);
            y = reshape(y(:), sz_prmt);
            y = ipermute(y,prmt);
          else
            y = reshape(y(:), [sz(k), numel(y)/sz(k)]);
            y = mtx.multMat(y);
          end
        end
        sz(k) = dims(k,1);
      end
      y = y(:);
    end
  end
  
  methods (Static, Access=private)
    function indcs = compIndcsNoClip(mtrx, trnsp)
      indcs = [];
      ind = mtrx{1}.indcsNoClip(trnsp);
      if isempty(ind)
        return
      end
      v = false(getNRows(1),1);
      v(ind) = true;
      for k=2:length(mtrx)
        ind = mtrx{k}.indcsNoClip(trnsp);
        if isempty(ind)
          return
        end
        w = false(getNRows(k),1);
        w(ind) = true;
        v = kron(v,w);
      end
      indcs = find(v);
      
      function nrw = getNRows(kk)
        if trnsp
          nrw = mtrx{kk}.nRows();
        else
          nrw = mtrx{kk}.nCols();
        end
      end
    end
%     function indcs = compIndcsNoClip(mtrx)
%       indcs = double(mtrx{1}.indcsNoClip());
%       nrwk = double(mtrx{1}.nRows());
%       nrwk1 = 1;
%       for k=2:length(mtrx)
%         nrwk1 = nrwk1 * nrwk;
%         nrwk = double(mtrx{k}.nRows());
%         
%         if ~isempty(indcs)
%           indcs1 = ones(nrwk,1) * (indcs-1)' * nrwk +...
%             (1:nrwk)' * ones(1,length(indcs));
%         else
%           indcs1 = [];
%         end
%         
%         indcs2 = double(mtrx{k}.indcsNoClip());
%         if ~isempty(indcs2)
%           indcs2 = indcs2 * ones(1,nrwk1) + ...
%             nrwk * ones(length(indcs2),1) * (0:(nrwk1-1));
%         else
%           indcs2 = [];
%         end
%         indcs = unique([indcs1(:); indcs2(:)]);
%       end
%       
%     end
  end
end
