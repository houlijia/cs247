classdef SensingMatrixCascade < SensingMatrixComposed
    %SensingMatrixCascade is a composition of several matrices.
    %   mtrx{1}*mtrx{2}*...
    
    properties
    end
    
    methods
       function obj = SensingMatrixCascade(varargin)
         % Constructor
         %   Input:
         %     mtrcs - a cell array of matrices. Each cell contains either a matrix
         %             object or struct specifying a matrix (a struct with fields
         %             'type' and 'args'). Alternatively mtrcs can be a struct
         %             specifying the matrrix.
            obj.setSensingMatrixCascade(varargin{:})
        end
        
        function set(obj, varargin)
          % Initialize the object
          %   Input:
          %     obj - this object
          %     mtrcs - a cell array of matrices. Each cell contains either a matrix
          %             object or struct specifying a matrix (a struct with fields
          %             'type' and 'args'). Alternatively mtrcs can be a struct
          %             specifying the matrrix.
          %     mtrcs - the matrices sequence
          varargin = parseInitArgs(varargin, {'mtrcs'});
            obj.setSensingMatrixCascade(varargin{:})
        end
        
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = doMultVec(obj, x)
            y = x;
            for k=length(obj.mtrx):-1:1
                y = obj.mtrx{k}.multVec(y);
            end
        end
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspVec(obj, x)
            y = x;
            for k=1:length(obj.mtrx)
                y = obj.mtrx{k}.multTrnspVec(y);
            end
        end
        
        % doMultMat - Multiply a matrix x with n_cols rows by the matrix and 
        % return a vector matrix of length n_rows.
        function y = doMultMat(obj, x)
            y = x;
            for k=length(obj.mtrx):-1:1
                y = obj.mtrx{k}.multMat(y);
            end
        end
        
        % doMultTrnspMat - Multiply a matrix x with n_rows rows by the transpose 
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = doMultTrnspMat(obj, x)
            y = x;
            for k=1:length(obj.mtrx)
                y = obj.mtrx{k}.multTrnspMat(y);
            end
        end
        

        % Computes the matlab matrix which corresponds to what this matrix
        % should be.
        function mtrx = doCompMatrix(obj)
          if ~obj.is_transposed
            mtrx = obj.mtrx{1}.getMatrix();
            for k=2:length(obj.mtrx)
              mtrx = mtrx * obj.mtrx{k}.getMatrix();
            end
          else
            mtx = obj.mtrx{1};
            mtx.transpose();
            mtrx = obj.mtrx{1}.getMatrix();
            mtx.transpose();
            for k=2:length(obj.mtrx)
              mtx = obj.mtrx{k};
              mtx.transpose();
              mtrx = mtx.getMatrix() * mtrx;
              mtx.transpose();
            end
          end
        end
    
        % Returns the sum of values of the measurement which contain DC value,
        % weighted by the ratio of the DC value to other components (in
        % terms of RMS), or 0 if there is no such measurement.
        %   Input:
        %     obj - this object
        %     msrs - the measurements vector
        % The value here is base on the last matrix only.
        function dc_val = getDC(obj,msrs)
            dc_val = obj.mtrx{1}.getDC(msrs);
        end
        
        function L = do_compSVD1(obj, complete)
          nm = numel(obj.mtrx);
          if obj.is_transposed
            obj.trasnpose();
            L = obj.do_compSVD1(complete);
            obj.transpose()
            L = obj.transposeSVD(L);
          elseif nm == 1
            L = obj.mtrx{1}.do_compSVD1(obj, complete);
          elseif isa(obj.mtrx{1},'SensingMatrixScaler') || ...
              isa(obj.mtrx{end},'SensingMatrixScaler')
            if isa(obj.mtrx{1},'SensingMatrixScaler')
              mt = obj.constructCascade(obj.mtrx(2:end));
              scl = obj.mtrx{1}.scaler();
            else
              mt = obj.constructCascade(obj.mtrx(1:(end-1)));
              scl = obj.mtrx{end}.scaler();
            end
            L = mt.do_compSVD1(complete);
            L = L.copyScale(abs(scl));
          elseif obj.mtrx{1}.getOrthoCol() && ~obj.getOrthoCol()
            mt = obj.constructCascade(obj.mtrx(2:end));
            L = mt.do_compSVD1(complete);
            if complete
              L = SensingMatrixDiag.constructDiag(L.getDiag(), ...
                obj.nRows(), obj.nCols());
            end
          elseif obj.mtrx{end}.getOrthoRow() && ~obj.getOrthoRow()
            % Create the transpose and use the previous case
            mtrcs = cell(size(obj.mtrx));
            for k=1:numel(mtrcs)
              mtrcs{k} = obj.mtrx{end-k+1}.copy();
              mtrcs{k}.transpose();
            end
            mt = obj.constructCascade(mtrcs);
            L = mt.do_compSVD3(complete);
            L = obj.transposeSVD(L);
          else
            L = obj.do_compSVD1@SensingMatrix();
          end
        end
    
        function [L,U,V] = do_compSVD3(obj, complete)
          nm = numel(obj.mtrx);
          if obj.is_transposed
            obj.trasnpose();
            [L,U,V] = obj.do_compSVD3(complete);
            obj.transpose()
            [L,U,V] = obj.transposeSVD(L,U,V);
          elseif nm == 1
            [L,U,V] = obj.mtrx{1}.do_compSVD3(complete);
          elseif isa(obj.mtrx{1},'SensingMatrixScaler') || ...
              isa(obj.mtrx{end},'SensingMatrixScaler')
            if isa(obj.mtrx{1},'SensingMatrixScaler')
              mt = obj.constructCascade(obj.mtrx(2:end));
              scl = obj.mtrx{1}.scaler();
            else
              mt = obj.constructCascade(obj.mtrx(1:(end-1)), complete);
              scl = obj.mtrx{end}.scaler();
            end
            [L,U,V] = mt.do_compSVD3(complete);
            L = L.copyScale(abs(scl));
            if scl < 0
              U = obj.constructCascade({U,SensingMatrixScaler(U.nCols(),-1)});
            end
          elseif obj.mtrx{1}.getOrthoCol() && ~obj.getOrthoCol()
            % Using the facts that if X and you are column orthogonal, so
            % is X*Y.  Furthermore, if X=U*L*V' then the diagonal of L (the
            % singular values) is constant and V is a squar permutation
            % matrix
            scl = obj.mtrx{1}.getExactNorm();
            nr = obj.nRows();
            nc = obj.nCols();
            if scl == 0       % Special case of a zero matrix
              if complete
                L = SensingMatrixDiag.constructDiag(obj.zeros(nc,1), nr, nc);
                U = SensingMatrixUnit(nr);
              else
                L = SensingMatrixScaler(nc, 0);
                U = SensingMatrixDiag.constructDiag(obj.ones(nc,1), nr, nc);
              end
              V = SensingMatrixUnit(nc);
            else  % common case
              nc1 = obj.mtrx{1}.nCols();
              nr1 = obj.mtrx{1}.nRows();
              mt1 = obj.constructCascade({obj.mtrx{1},SensingMatrixScaler(nc1,1/scl)});
              mt2 = obj.constructCascade([obj.mtrx(2:end);{SensingMatrixScaler(nc,scl)}]);
              [L2,U2,V2] = mt2.do_compSVD3(complete);
              V = V2;
              W = obj.constructCascade({mt1,U2});
              % obj = W*L2*V2' and W'*W = I. Therefore, if ~complete or if
              % nc1==nr1 this is the SVD. 
              if ~complete || nc1==nr1
                L = L2;
                U = W;
              else
                % Let WN be the null space of W, so that the matrix [W,WN]
                % is square and orthonormal. Then W=[W,WN]*J, where J=[I;0]'
                % is a tall diagonal matrix with a diagonal entries of one.
                % Therefore, 
                %   obj = W*L2*V2'= [W,WN]*(J*L2)*V2'
                % J*L2 is a rectangular diagonal matrix, whose diagonal is
                % the same as L2, possibly extended by zeros.
                % Therefore, U=[W,WN], L=(J*L2) V=V2.
                %
                % In order to find WN we compute complete SVD of W,
                %   W=U1*L1*V1'
                % L1 must be a nr1Xnc1 diagonal matrix with ones in the
                % diagonal. Therefore any of the nc1+1,...,nr1 right
                % columns of W is in the null space.
                % 
                [~,U1,~] = W.do_compSVD3(complete);
                % Since SensingMatrixSelectRange andSensingMatrixConcat 
                % work on rows, everything is done in transpose and then 
                % transposed back.
                S = SensingMatrixSelectRange.constructSelectRange(nc1+1,nr1,nr1);
                U1.transpose();
                WN = obj.constructCascade({S,U1});
                W.transpose();
                U = SensingMatrixConcat({W,WN});
                U.transpose();   % transpose back
                U.setOrthoCol(true);
                
                dg = L2.getDiag();
                L = SensingMatrixDiag.constructDiag(dg, nr, nc);
              end
            end
          elseif obj.mtrx{end}.getOrthoRow() && ~obj.getOrthoRow()
            % Create the transpose and use the previous case
            mtrcs = cell(size(obj.mtrx));
            for k=1:numel(mtrcs)
              mtrcs{k} = obj.mtrx{end-k+1}.copy();
              mtrcs{k}.transpose();
            end
            mt = obj.constructCascade(mtrcs);
            [L,U,V] = mt.do_compSVD3(complete);
            [L,U,V] = obj.transposeSVD(L,U,V);
          else
            [L,U,V] = obj.do_compSVD3@SensingMatrix(complete);
          end
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
        
        function dg = getDiag(obj)
          dg = obj.mtrx{1}.getDiag();
          for k=2:length(obj.mtrx)
            dg1 = obj.mtrx{k}.getDiag();
            mn = min(length(dg),length(dg1));
            dg = dg(1:mn) .* dg1(1:mn);
          end
        end
        
        function mtx = copyScale(obj, scl)
          mtxs = [obj.mtx(:); {SensingMatrixScaler(scl, obj.nCols())}];
          mtx = obj.constructCascade(mtxs);
          if obj.is_transposed
            mtx.transpose();
          end
        end            
        
        function val = cmpExactNorm(obj)
          val = 1;
          mtxs = obj.mtrx;
          is_scaler = false(size(mtxs));
          for k=1:length(mtxs)
            if isa(mtxs{k},'SensingMatrixScaler')
              val = val * mtxs{k}.getExactNorm();
              is_scaler(k) = true;
            end
          end
          mtxs(is_scaler) = [];
          b = 1;
          e = length(mtxs);
          while b<=length(mtxs)
            if ~mtxs{b}.getOrtho()
              break;
            end
            val = val * mtxs{b}.getExactNorm();
            b = b+1;
          end
          while e>=b
            if ~mtxs{e}.getOrtho()
              break;
            end
            val = val * mtxs{e}.getExactNorm();
            e = e-1;
          end
          
          mtxs = mtxs(b:e);
          
          if ~isempty(mtxs)
            m = obj.constructCascade(mtxs);
            if isa(m, 'SensingMatrixCascade')
              val = val * m.cmpExactNorm@SensingMatrix();
            else
              val = val * m.cmpExactNorm();
            end
          end
        end
    end

    methods (Access=protected)
        function setSensingMatrixCascade(obj, mtrcs)
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
          
          if ~isempty(obj.mtrx)
            obj.setIndcsNoClip(obj.mtrx{1}.indcsNoClip(), false);
            obj.setIndcsNoClip(obj.mtrx{end}.indcsNoClip(true), true);
          end
        end

        function [ncl, nrw, orth, psd_orth] = compDim(~, mtrcs)
            nrw = mtrcs{1}.nRows();
            ncl = mtrcs{end}.nCols();
            orth = struct('col', mtrcs{1}.getOrthoCol(),...
              'row',  mtrcs{1}.getOrthoRow());
            for k=2:length(mtrcs)
                if mtrcs{k}.nRows() ~= mtrcs{k-1}.nCols()
                    error('mtrcs{%d}.nRows()=%d ~= mtrcs{%d}.nCols()=%d',...
                        k, mtrcs{k}.nRows(), k-1, mtrcs{k-1}.nCols());
                end
                orth.col = orth.col && mtrcs{k}.getOrthoCol();
                orth.row = orth.row && mtrcs{k}.getOrthoRow();
            end
            if k==1
              psd_orth = struct('row', mtrcs{1}.getPsdOrthoRow(), 'col', ...
                mtrcs{1}.getPsdOrthoCol());
            else
              psd_orth = orth;
            end
        end
 
        % This function compute an upper bound, not the actual norm
        function nrm = compNorm(obj)
            n_mtrx = length(obj.mtrx);
%             nrm_aa = obj.toCPUFloat(zeros(n_mtrx,1));
%             for k=1:n_mtrx
%               nrm_aa(k) = obj.mtrx{k}.norm();
%             end
            nrm_aa = arrayfun(@(k) obj.mtrx{k}.norm(), (1:n_mtrx)');
            nrm = prod(nrm_aa);
        end
    end

    methods(Static)
      function mtx = construct(mtrcs)
        mtx = SensingMatrixCascade.constructCascade(mtrcs);
      end
      
      function mtx = constructCascade(mtrcs)
        
        function elmnts = get_elmnts(mt)
          if iscell(mt)
            elmnts = cell(1,length(mt));
            for i=1:length(mt)
              elmnts{i} = get_elmnts(mt{i});
            end
            elmnts = horzcat(elmnts{:});
            return
          elseif ~isa(mt,'SensingMatrixCascade')
            elmnts = {mt.copy()};
            return
          end
          elmnts = cell(1,length(mt.mtrx));
          if ~mt.is_transposed
            for i=1:length(elmnts)
              elmnt = mt.mtrx{i}.copy();
              elmnts{i} = get_elmnts(elmnt);
            end
          else
            for i=1:length(elmnts)
              elmnt = mt.mtrx{i}.copy();
              elmnt.transpose();
              elmnts{end+1-i} = get_elmnts(elmnt);
            end
          end
          elmnts = horzcat(elmnts{:});
        end
        
        for k=1:(length(mtrcs)-1)
          if mtrcs{k}.nCols() ~= mtrcs{k+1}.nRows()
            error('mtrcs(%d) is %dX%d but mtrcs(%d) is %dX%d', ...
              k, mtrcs{k}.nRows(), mtrcs{k}.nCols(), ...
              k+1, mtrcs{k+1}.nRows(), mtrcs{k+1}.nCols());
          end
        end
        mtxs = get_elmnts(mtrcs);
        mtrx = cell(size(mtxs));
        nr1 = mtxs{1}.nRows();
        scl = 1;
        dg_indx = 0;
        mt_indx = 0;
        is_sclr = false(size(mtrx));
        for k=1:length(mtxs)
          if isa(mtxs{k}, 'SensingMatrixScaler')
            scl = scl * mtxs{k}.scaler();
            is_sclr(k) = true;
          elseif ~dg_indx && isa(mtxs{k}, 'SensingMatrixDiag')
            dg_indx = k;
          elseif ~dg_indx && ~mt_indx && isa(mtxs(k), 'SensingMatrixMatlab')
            mt_indx = k;
          end
        end
        if scl ~= 1 
          if dg_indx
            mtxs{dg_indx} = SensingMatrixDiag(scl * mtxs{dg_indx}.getDiag());
            scl = 1;
          elseif mt_indx
            mtxs{mt_indx} = SensingMatrixMatlab(scl*mtxs{mt_indx}.getMatrix());
            scl = 1;
          end
        end
        mtxs(is_sclr) = [];
        n_mtxs = length(mtxs);
        n_mtrx = 1;
        if scl == 1
          mtrx{n_mtrx} = SensingMatrixUnit(nr1);
        else
          mtrx{n_mtrx} = SensingMatrixScaler(nr1, scl);
        end
        
        for k=1:n_mtxs
          % simplify notation (since those are handles)
          mtr = mtrx{n_mtrx};
          mts = mtxs{k};
          
          if isa(mtr, 'SensingMatrixUnit')
            mtrx{n_mtrx} = mts;
          elseif isa(mts, 'SensingMatrixDiag') && ...
              isa(mtr, 'SensingMatrixDiag')
            mtrx{n_mtrx} = SensingMatrixDiag.constructDiag(...
              mts.getDiag() .*  mtr.getDiag());
          elseif isa(mts, 'SensingMatrixSelect') && ...
              isa(mtr, 'SensingMatrixSelect') && ...
              (mts.is_transposed == mtr.is_transposed ||...
              mtr.isPermutation() || mts.isPermutation())
            indx1 = mtr.getIndices();
            indx2 = mts.getIndices();
            is_trnsp = mts.is_transposed;
            if mts.is_transposed ~= mtr.is_transposed
              if mtr.isPermutation()
                indx1 = inv_permutation(indx1);
              else
                indx2 = inv_permutation(indx2);
                is_trnsp = ~is_trnsp;
              end
            end
            
            if ~is_trnsp
              indx = indx2(indx1);
              nc = mts.nCols();
            else
              indx = indx1(indx2);
              nc = mtr.nRows();
            end
            mtrx{n_mtrx} = ...
              SensingMatrixSelect.construct(indx, nc, mts.is_transposed);
          elseif (isa(mtr, 'SensingMatrixMatlab') && isa(mts, 'SensingMatrixMatlab')) &&...
              ((mtr.nRows()*mts.nCols() <= mtr.nRows()*mtr.nCols() + mts.nRows*mts.nCols()) || ...
              (mts.nCols() <= mtr.nCols() && mts.isSparse()) || ...
              (mtr.nRows() <= mts.nRows() && mtr.isSparse()) || ...
              (mtr.isSparse() && mts.isSparse()))
            mtrx{n_mtrx} = SensingMatrixMatlab(mtr.getMatrix() * mts.getMatrix());
          elseif isa(mts, 'SensingMatrixMatlab') && (isa(mtr,'SensingMatrixDiag') ||...
              (isa(mtr,'SensingMatrixSelect') && ...
              (mts.isSparse() || mtr.nRows() <= mtr.nCols())))
            mt = mts.getMatrix();
            mt = mtr.multMat(mt);
            mtrx{n_mtrx} = SensingMatrixMatlab(mt);
          elseif isa(mtr, 'SensingMatrixMatlab') &&(isa(mts,'SensingMatrixDiag') ||...
              (isa(mts,'SensingMatrixSelect') && ...
              (mtr.isSparse() || mts.nRows() <= mts.nCols())))
            mt = mtr.getMatrix();
            mt = mts.multTrnspMat(mt');
            mtrx{n_mtrx} = SensingMatrixMatlab(mt');
          elseif chk_strctr(mtr, mts, 'SensingMatrixBlkDiag')
            nrx = length(mtr.mtrx);
            mt = cell(nrx,1);
            mtr = do_trnsp(mtr);
            mts = do_trnsp(mts);
            for j=1:nrx
              mt{j} = SensingMatrixCascade.constructCascade(...
                {mtr.mtrx{j}, mts.mtrx{j}});
            end
            mtrx{n_mtrx} = SensingMatrixBlkDiag(mt);
          elseif chk_strctr(mtr, mts, 'SensingMatrixKron')
            nrx = length(mtr.mtrx);
            mt = cell(nrx,1);
            mtr = do_trnsp(mtr);
            mts = do_trnsp(mts);
            for j=1:nrx
              mt{j} = SensingMatrixCascade.constructCascade(...
                {mtr.mtrx{j}, mts.mtrx{j}});
            end
            mtrx{n_mtrx} = SensingMatrixKron(mt);
          else
            n_mtrx = n_mtrx+1;
            mtrx{n_mtrx} = mts;
          end
          
          if n_mtrx>1 && isa(mtrx{n_mtrx},'SensingMatrixUnit')
            n_mtrx = n_mtrx-1;
          end
        end
        if n_mtrx > 1
          mtx = SensingMatrixCascade(mtrx(1:n_mtrx));
        else
          mtx = mtrx{1};
        end
        
        function mtch = chk_strctr(m1,m2, cls)
          mtch = false;
          if ~isa(m1, cls) || ~isa(m2, cls)
            return
          elseif length(m1.mtrx) ~= length(m2.mtrx)
            return
          end
          for jj=1:length(m1.mtrx)
            if ~m1.is_transposed
              m1c = m1.mtrx{jj}.nCols();
            else
              m1c = m1.mtrx{jj}.nRows();
            end
            if ~m2.is_transposed
              m2r = m2.mtrx{jj}.nRows();
            else
              m2r = m2.mtrx{jj}.nCols();
            end
            if m1c ~= m2r
              return
            end
          end
          mtch = true;
        end
        
        function mt = do_trnsp(mt)
          if ~mt.is_transposed
            return
          end
          mt = mt.copy();
          mt.transpose();
          for jj=1:length(mt.mtrx)
            mt.mtrx{jj}.transpose();
          end
        end
      end
    end
end

