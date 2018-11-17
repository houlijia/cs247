classdef SensingMatrixBlkDiag < SensingMatrixComposed
  %SensingMatrixBlkDiag Concatenation of several sensing matrices (the
  %measurement vector is the concatenation of the measurements vectors of
  %all matrices)
  %
  
  
  methods
    function obj = SensingMatrixBlkDiag(varargin)
      % Constructor can have 0 or 1 arguments.
      %   Input
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      obj.setSensingMatrixBlkDiag(varargin{:})
    end
    
    function set(obj, varargin)
      %   Input
      %     obj - this object
      %     mtrcs - a cell array of matrices. Each cell contains either a matrix
      %             object or struct specifying a matrix (a struct with fields
      %             'type' and 'args'). Alternatively mtrcs can be a struct
      %             specifying the matrrix.
      varargin = parseInitArgs(varargin, {'mtrcs'});
      obj.setSensingMatrixBlkDiag(varargin{:})
    end
    
    % multVec - Multiply a vector x of length n_cols by the matrix and return
    % a vector y of length n_rows.
    function y = doMultVec(obj, x)
      vecs = arrayfun(@mult_vec_k, (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
      %             y = zeros(obj.n_rows, 1);
      %             xbgn = 1;
      %             ybgn = 1;
      %             for k=1:length(obj.mtrx)
      %                 new_xbgn = xbgn + obj.mtrx{k}.nCols();
      %                 new_ybgn = ybgn + obj.mtrx{k}.nRows();
      %                 y(ybgn:(new_ybgn-1)) = ...
      %                     obj.mtrx{k}.multVec(x(xbgn:(new_xbgn-1)));
      %                 xbgn = new_xbgn;
      %                 ybgn = new_ybgn;
      %             end
      
      function cv = mult_vec_k(k)
        cv = {obj.mtrx{k}.multVec(x(obj.mtrx_bgn(k,1):obj.mtrx_end(k,1)))};
      end
    end
    
    % multTrnspVec - Multiply a vector x of length n_rows by the transpose
    % of the matrix and return A vector y of length n_cols.  The result
    % may be scaled (see below).
    function y = doMultTrnspVec(obj, x)
      vecs = arrayfun(...
        @(k) {obj.mtrx{k}.multTrnspVec(x(obj.mtrx_bgn(k,2):obj.mtrx_end(k,2)))},...
        (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
      %             y = zeros(obj.n_cols, 1);
      %             xbgn = 1;
      %             ybgn = 1;
      %             for k=1:length(obj.mtrx)
      %                 new_xbgn = xbgn + obj.mtrx{k}.nRows();
      %                 new_ybgn = ybgn + obj.mtrx{k}.nCols();
      %                 y(ybgn:(new_ybgn-1)) = ...
      %                     obj.mtrx{k}.multTrnspVec(x(xbgn:(new_xbgn-1)));
      %                 xbgn = new_xbgn;
      %                 ybgn = new_ybgn;
      %             end
    end
    
    function y = doMultMat(obj, x)
      vecs = arrayfun(...
        @(k) {obj.mtrx{k}.multMat(x(obj.mtrx_bgn(k,1):obj.mtrx_end(k,1),:))},...
        (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
    end
    
    function y = doMultTrnspMat(obj, x)
      vecs = arrayfun(...
        @(k) {obj.mtrx{k}.multTrnspMat(x(obj.mtrx_bgn(k,2):obj.mtrx_end(k,2),:))},...
        (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
    end
    
    % Multiply a matrix or a vector by the matrix whose entries are the
    % absolute values of the entries of this matrix.
    function y = multAbs(obj, x)
      if obj.is_transposed
        idx = 2;
        for k=1:length(obj.mtrx)
          obj.mtrx{k}.transpose();
        end
      else
        idx = 1;
      end
      
      vecs = arrayfun(...
        @(k) {obj.mtrx{k}.multAbs(x(obj.mtrx_bgn(k,idx):obj.mtrx_end(k,idx)))},...
        (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
      
      if obj.is_transposed
        for k=1:length(obj.mtrx)
          obj.mtrx{k}.transpose();
        end
      end
      %
      %           y = zeros(obj.nRows(), size(x,2));
      %           xbgn = 1;
      %           ybgn = 1;
      %           for k=1:length(obj.mtrx)
      %             if obj.is_transposed
      %               obj.mtrx{k}.transpose();
      %             end
      %
      %             new_xbgn = xbgn + obj.mtrx{k}.nCols();
      %             new_ybgn = ybgn + obj.mtrx{k}.nRows();
      %             y(ybgn:(new_ybgn-1),:) = ...
      %               obj.mtrx{k}.multAbs(x(xbgn:(new_xbgn-1)));
      %             xbgn = new_xbgn;
      %             ybgn = new_ybgn;
      %
      %             if obj.is_transposed
      %               obj.mtrx{k}.transpose();
      %             end
      %           end
    end
    
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(obj,y)
      if obj.is_transposed
        idx = 1;
        for k=1:length(obj.mtrx)
          obj.mtrx{k}.transpose();
        end
      else
        idx = 2;
      end
      
      vecs = arrayfun(...
        @(k) {obj.mtrx{k}.normalizeMsrs(y(obj.mtrx_bgn(k,idx):obj.mtrx_end(k,idx)))},...
        (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
      
      if obj.is_transposed
        for k=1:length(obj.mtrx)
          obj.mtrx{k}.transpose();
        end
      end
      
      %           ybgn = 1;
      %           for k=1:length(obj.mtrx)
      %             mtx = obj.mtrx{k};
      %             if obj.is_transposed
      %               mtx.transpose();
      %             end
      %             yend = ybgn + mtx.nRows() - 1;
      %             y(ybgn:yend) = mtx.normalizeMsrs(y(ybgn:yend));
      %             ybgn = yend +1;
      %             if obj.is_transposed
      %               mtx.transpose();
      %             end
      %           end
    end
    
    % undo the operation of normalizeMsrs
    function y=deNormalizeMsrs(obj,y)
      if obj.is_transposed
        idx = 1;
        for k=1:length(obj.mtrx)
          obj.mtrx{k}.transpose();
        end
      else
        idx = 2;
      end
      
      vecs = arrayfun(...
        @(k) {obj.mtrx{k}.deNormalizeMsrs(y(obj.mtrx_bgn(k,idx):obj.mtrx_end(k,idx)))},...
        (1:numel(obj.mtrx)));
      y = vertcat(vecs{:});
      
      if obj.is_transposed
        for k=1:length(obj.mtrx)
          obj.mtrx{k}.transpose();
        end
      end
      
      %           ybgn = 1;
      %           for k=1:length(obj.mtrx)
      %             mtx = obj.mtrx{k};
      %             if obj.is_transposed
      %               mtx.transpose();
      %             end
      %             yend = ybgn + mtx.nRows() - 1;
      %             y(ybgn:yend) = mtx.deNormalizeMsrs(y(ybgn:yend));
      %             ybgn = yend +1;
      %             if obj.is_transposed
      %               mtx.transpose();
      %             end
      %           end
    end
        
    % Computes the matlab matrix which corresponds to what this matrix
    % should be.
    function mtrx = doCompMatrix(obj)
      r_bgn = 0;
      c_bgn = 0;
      n_mtrx = length(obj.mtrx);
      if obj.isSparse() && ~obj.use_gpu && ~obj.use_single
        rr = cell(n_mtrx,1); % row indices
        cc = cell(n_mtrx,1); % column indices
        vv = cell(n_mtrx,1); % values
        n_ttl = 0;
        for k=1:n_mtrx
          [r,c,v] = find(obj.mtrx{k}.getMatrix());
          rr{k} = r + r_bgn;
          cc{k} = c + c_bgn;
          vv{k} = v;
          r_bgn = r_bgn + double(obj.mtrx{k}.nRows());
          c_bgn = c_bgn + double(obj.mtrx{k}.nCols());
          n_ttl = n_ttl + length(r);
        end
        r = double(vertcat(rr{:}));
        c = double(vertcat(cc{:}));
        v = double(vertcat(vv{:}));
        if obj.is_transposed
          mtrx = sparse(c,r,v,double(obj.n_cols),double(obj.n_rows));
        else
          mtrx = sparse(r,c,v,double(obj.n_rows),double(obj.n_cols));
        end
      else
        mtrx = obj.zeros(obj.nRows(), obj.nCols());
        for k=1:n_mtrx
          mtx = obj.mtrx{k}.getMatrix();
          if obj.is_transposed
            mtx = mtx';
          end
          r_end = r_bgn + size(mtx,1);
          c_end = c_bgn + size(mtx,2);
          mtrx(r_bgn+1:r_end, c_bgn+1:c_end) = mtx;
          r_bgn = r_end;
          c_bgn = c_end;
        end
      end
    end
    
    
    % Returns the sum of values of the measurement which contain DC value,
    % weighted by the ratio of the DC value to other components (in
    % terms of RMS), or 0 if there is no such measurement.
    %   Input:
    %     obj - this object
    %     msrs - the measurements vector
    function dc_val = getDC(obj,msrs)
      dc_val = 0;
      xbgn = 1;
      for k=1:length(obj.mtrx)
        mtx = obj.mtrx{k};
        if ~mtx.nRows()
          continue;
        end
        new_xbgn = xbgn + mtx.nRows();
        dcv = mtx.getDC(msrs(xbgn:new_xbgn-1));
        if isempty(dcv)
          dc_val = [];
          return;
        end
        dc_val = dc_val + dcv;
        xbgn = new_xbgn;
      end
    end
    
    function dg = getDiag(obj)
      dgs = cell(size(obj.mtrx));
      for k = 1:numel(obj.mtrx)
        dgs{k} = obj.mtrx{k}.getDiag();
      end
      dg = vertcat(dgs{:});
    end
    
    % Set an exact value for norm(). It can be computationally heavy
    function val = cmpExactNorm(obj)
      n_mtrx = length(obj.mtrx);
      nrm_aa = arrayfun(@(k) obj.mtrx{k}.getExactNorm(), (1:n_mtrx)');
%       nrm_aa = obj.toCPUFloat(zeros(n_mtrx,1));
%       for k=1:n_mtrx
%         nrm_aa(k) = obj.mtrx{k}.getExactNorm();
%       end
      val = max(nrm_aa);
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
        L{k} = L{k}.getDiag();
        L{k} = L{k}(:);
      end
      L = obj.compSVDfromBlks(L);
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
      L = cell(nm,1);
      U = cell(nm,1);
      V = cell(nm,1);
      for k=1:nm
        [L{k},U{k},V{k}] = obj.mtrx{k}.do_compSVD3(complete);
      end
      [L,U,V] = obj.compSVDfromBlks(L,U,V);
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
      % construct a block diag matrix. mtrcs can be either a cell array of
      % matrics, or a struct with a field 'mtrcs', whose value is a cell array
      % of matrices.
      if isstruct(mtrcs)
        mtrcs = mtrcs.mtrcs;
      end
      mtrx = SensingMatrixBlkDiag.constructBlkDiag(mtrcs);
    end
    
    function mtrx = constructBlkDiag(mtrcs)
      function elmnts = get_elmnts(mt)
        if iscell(mt)
          elmnts = cell(1,length(mt));
          for i=1:length(mt)
            elmnts{i} = get_elmnts(mt{i});
          end
          elmnts = horzcat(elmnts{:});
          return
        elseif ~isa(mt, 'SensingMatrixBlkDiag')
          elmnts = {mt};
          return
        end
        elmnts = cell(1,length(mt.mtrx));
        if ~mt.is_transposed
          for i=1:length(elmnts)
            elmnt = mt.mtrx{i};
            elmnts{i} = get_elmnts(elmnt);
          end
        else
          for i=1:length(elmnts)
            matched = false;
            for i1=1:(i-1)
              if isEqual(mt.mtrx{i}, mt.mtrx{i1})
                elmnts{i} = elmnts{i1};
                matched = true;
                break;
              end
            end
            if ~matched
              elmnt = mt.mtrx{i}.copy();
              elmnt.transpose();
              elmnts{i} = get_elmnts(elmnt);
            end
          end
        end
        elmnts = horzcat(elmnts{:});
      end
      
      mtxs = get_elmnts(mtrcs);
      n_mtxs = length(mtxs);
      
      % Combine identical matrices into Kronecker products
      prev_n_mtxs = n_mtxs;
      while n_mtxs > 1
        k=0;  % Index to put into
        b=1;  % Beginning of sequence of equal matrices
        for j=1:n_mtxs
          if j==b || mtxs{b}.isEqual(mtxs{j})
            if j==n_mtxs
              k=k+1;
              if j>b
                mtxs{k} = comp_kron(mtxs{b}, j-b+1);
              else
                mtxs{k} = mtxs{b};
              end
            else
              continue
            end
          else
            k = k+1;
            if j-1>b
              mtxs{k} = comp_kron(mtxs{b}, j-b);
            else
              mtxs{k} = mtxs{b};
            end
            if j==n_mtxs
              k = k+1;
              mtxs{k} = mtxs{j};
            else
              b = j;
            end
          end
        end
        mtxs = mtxs(1:k);
        n_mtxs = k;
        if prev_n_mtxs == n_mtxs
          break
        end
        prev_n_mtxs = n_mtxs;
      end
      
      function mtx = comp_kron(mtx, cnt)
        indcs_no_clip = mtx.indcsNoClip();
        n_r = double(mtx.nRows());
        
        mtx = SensingMatrixKron.construct({SensingMatrixUnit(cnt), mtx});
        
        if isa(mtx, 'SensingMatrixKron')
          nclp = double(indcs_no_clip(:));
          nclp = nclp * ones(1,cnt) + n_r * ones(length(nclp),1)*(0:cnt-1);
          nclp = cast(nclp(:), 'like', indcs_no_clip);
          mtx.setIndcsNoClip(nclp);
        end
      end
      
      mtrx = cell(size(mtxs));
      n_mtrx = 1;
      mtrx{n_mtrx} = mtxs{1};
      
      for k=2:n_mtxs
        if isa(mtxs{k}, 'SensingMatrixUnit') && ...
            isa(mtrx{n_mtrx},'SensingMatrixUnit')
          ncl = mtxs{k}.nCols() + mtrx{n_mtrx}.nCols();
          mtrx{n_mtrx} = SensingMatrixUnit(ncl);
        elseif isa(mtxs{k}, 'SensingMatrixScaler') && ...
            isa(mtrx{n_mtrx},'SensingMatrixScaler') && ...
            mtxs{k}.scaler() == mtrx{n_mtrx}.scaler()
          ncl = mtxs{k}.nCols() + mtrx{n_mtrx}.nCols();
          mtrx{n_mtrx} = SensingMatrixScaler(ncl, mtrx{n_mtrx}.scaler());
        elseif isa(mtxs{k}, 'SensingMatrixDiag') && ...
            isa(mtrx{n_mtrx},'SensingMatrixDiag') && ...
            ~isa(mtxs{k}, 'SensingMatrixScaler') && ...
            ~isa(mtrx{n_mtrx},'SensingMatrixScaler')
          dg = [mtrx{n_mtrx}.getDiag(); mtxs{k}.getDiag()];
          mtrx{n_mtrx} = SensingMatrixDiag(dg);
        else
          n_mtrx = n_mtrx + 1;
          mtrx{n_mtrx} = mtxs{k};
        end
      end
      
      for k=1:n_mtrx
        mtrx{k} = mtrx{k}.copy();
      end
      if n_mtrx > 1
        mtrx = SensingMatrixBlkDiag(mtrx(1:n_mtrx));
      else
        mtrx = mtrx{1};
      end
    end
    
    % Compute SVD of a Block diagonal matrix from the SVDs of its
    % nT diagonal blocks.
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
    function [L,U,V] = compSVDfromBlks(LT,UT,VT,eps)
      if nargin < 4
        eps = 0;
        do_chkSVD = false;
        if nargin < 3 || nargout < 3
          VT = [];  UT = []; U = []; V = [];
        end
      else
        do_chkSVD = true;
      end
      
      nm = numel(LT);
      
      % check complete
      if isempty(UT)
        complete = false;
        for k=1:nm
          if LT{k}.nRows() ~= LT{k}.nCols()
            complete = true;
            break
          end
        end
      else
        complete = true;
        for k=1:nm
          if UT{k}.nRows() ~= UT{k}.nCols() || VT{k}.nRows() ~= VT{k}.nCols()
            complete = false;
            break
          end
        end
      end
      
      onr = 0;
      onc = 0;
      L = cell(nm,1);
      for k=1:nm
        if complete
          onr = onr + LT{k}.nRows();
          onc = onc + LT{k}.nCols();
        end
        L{k} = LT{k}.getDiag();
        L{k} = L{k}(:);
      end
      dg = vertcat(L{:});
      if ~complete
        L = SensingMatrixDiag.constructDiag(dg);
      else
        L = SensingMatrixDiag.constructDiag(dg, onr, onc);
      end
      
      if isempty(UT)
        L = SensingMatrix.sortSVD(L, [], [], eps);
        if do_chkSVD
          chkSVD(L, [], [], 2*eps);
        end
        return
      end
      
      if complete
        U = cell(nm,1);
        V = cell(nm,1);
        UN = cell(nm,1);
        VN = cell(nm,1);
        for k=1:nm
          [U{k},UN{k}] = split_cols(UT{k},LT{k}.nCols());
          [V{k},VN{k}] = split_cols(VT{k},LT{k}.nRows());
        end
        U = join_cols(U,UN);
        V = join_cols(V,VN);
      else
        U = SensingMatrixBlkDiag.constructBlkDiag(UT);
        V = SensingMatrixBlkDiag.constructBlkDiag(VT);
      end
      
      [L,U,V] = SensingMatrix.sortSVD(L,U,V,eps);
      
      if do_chkSVD
        MT = cell(nm,1);
        for k=1:nm
          MT{k} = SensingMatrix.constructSVD(LT{k},UT{k},VT{k});
        end
        M = SensingMatrixBlkDiag.constructBlkDiag(MT);
        SensingMatrix.chkSVD(L, U, V, 2*eps, M, complete);
      end
      
      function [B,BN] = split_cols(A, nc)
        nr = A.nRows();
        nc = min(nc,nr);
        if isa(A, 'SensingMatrixMatlab')
          A = A.getMatrix();
          B = SensingMatrixMatlab(A(:,1:nc));
          if nc<nr
            BN = SensingMatrixMatlab(A(:,nc+1:nr));
          else
            BN = SensingMatrixSelect([],nr);
            BN.transpose();
          end
        else
          S = SensingMatrixSelectRange.constructSelectRange(1,nc,nr);
          S.transpose();
          B = SensingMatrixCascade.constructCascade({A,S});
          S = SensingMatrixSelectRange.constructSelectRange(nc+1,nr,nr);
          S.transpose();
          if nc<nr
            BN = SensingMatrixCascade.constructCascade({A,S});
          else
            BN = S;
          end
        end
      end
      
      function A = join_cols(B,BN)
        B = SensingMatrixBlkDiag.constructBlkDiag(B);
        BN = SensingMatrixBlkDiag.constructBlkDiag(BN);
        if BN.nCols == 0
          A = B;
          return
        end
        B.transpose();
        BN.transpose();
        A = SensingMatrixConcat({B,BN});
        A.setOrthoRow(true);
        A.transpose();
      end
    end
    
  end
  
  properties (Access=private)
    % Begin and end index of the input vector components corresponding to each
    % component matrix. These are Nx2 matrices where the first column is
    % for the normal matrix and the second is for transposed matrix.
    mtrx_bgn;
    mtrx_end;
  end
  
  methods (Access=protected)
    %   Input
    %     obj - this object
    %     mtrcs - a cell array of matrices. Each cell contains either a matrix
    %             object or struct specifying a matrix (a struct with fields
    %             'type' and 'args'). Alternatively mtrcs can be a struct
    %             specifying the matrrix.
    function setSensingMatrixBlkDiag(obj, mtrcs)
      if nargin == 1
        sm_args = {};
      else
        sm_args = {mtrcs};
      end
      obj.setSensingMatrixComposed(sm_args{:});
      obj.setSparse(true);
      mtrx_len = obj.zeros(length(obj.mtrx),2);
      mtrx_len(:,1) = ...
        arrayfun(@(k) obj.mtrx{k}.nCols(), (1:numel(obj.mtrx))');
      mtrx_len(:,2) = ...
        arrayfun(@(k) obj.mtrx{k}.nRows(), (1:numel(obj.mtrx))');
      
      obj.mtrx_end = obj.toIndex(cumsum(mtrx_len));
      obj.mtrx_bgn = obj.toIndex(1 + [0,0; obj.mtrx_end(1:end-1,:)]);
      
      function indcell = get_indcs(k, trp)
        indcell = obj.mtrx{k}.indcsNoClip(trp) + obj.mtrx_bgn(k,2-trp) - 1;
        indcell = {indcell};
      end
      
      for trnsp = 0:1
        indcs = arrayfun(@(k) get_indcs(k,trnsp), (1:numel(obj.mtrx))');
        indcs = vertcat(indcs{:});
        obj.setIndcsNoClip(indcs, trnsp);
      end
    end
    
    function mtx = create(obj, args)
      mtx = obj.construct(args);
    end
    
    function [ncl, nrw, orth, psd_orth] = compDim(~, mtrcs)
      nrw = mtrcs{1}.nRows();
      ncl = mtrcs{1}.nCols();
      orth = struct('col', mtrcs{1}.getOrthoCol(), 'row',...
        mtrcs{1}.getOrthoRow());
      psd_orth =  struct('col', mtrcs{1}.getPsdOrthoCol(), 'row',...
        mtrcs{1}.getPsdOrthoRow());
      
      for k=2:length(mtrcs)
        ncl = ncl + mtrcs{k}.nCols();
        nrw = nrw + mtrcs{k}.nRows();
        orth.col = orth.col && mtrcs{k}.getOrthoCol();
        orth.row = orth.row && mtrcs{k}.getOrthoRow();
        psd_orth.col = psd_orth.col && mtrcs{k}.getPsdOrthoCol();
        psd_orth.row = psd_orth.row && mtrcs{k}.getPsdOrthoRow();
      end
      
      if length(mtrcs) > 1 && (orth.col || orth.row)
        nrms = arrayfun(@(kk) mtrcs{kk}.getExactNorm(), 1:length(mtrcs));
        if ~all(nrms(2:end) == nrms(1))
          orth.col = false;
          orth.row = false;
        end
      end
    end
    
    % Compute norm.
    function nrm = compNorm(obj)
      n_mtrx = length(obj.mtrx);
%       nrm_aa = obj.toCPUFloat(zeros(n_mtrx,1));
%       for k=1:n_mtrx
%         nrm_aa(k) = obj.mtrx{k}.norm();
%       end
      nrm_aa = arrayfun(@(k) obj.mtrx{k}.norm(), (1:n_mtrx)');
      nrm = max(nrm_aa);
    end
    
    function setUseGpu(obj,val)
      obj.setUseGpu@SensingMatrixComposed(val);
      obj.mtrx_bgn = obj.toIndex(obj.mtrx_bgn);
      obj.mtrx_end = obj.toIndex(obj.mtrx_end);
    end
    
  end
end

