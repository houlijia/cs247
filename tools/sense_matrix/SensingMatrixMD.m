classdef SensingMatrixMD < SensingMatrixCascade
  %SensingMatrixMD is a multi-dimensional sensing matrix
  %
  % The multidimensional sensing matrix is created by unidimensional sensing
  % matrices, one for each dimension, which are combined by Kronecker product
  % (using SensingMatrixKron). After the multiplication, the output is sorted
  % out using a zigzag ordering.
  
  properties(SetAccess=protected)
    dims;
    linear_indx;
    sbs_indx;
  end
  
  properties(Access=private)
    indcs_no_clip = -1;
    indcs_no_clip_trnsp = -1;
  end
  
  methods
    function obj = SensingMatrixMD(opts)
      % Constructor
      %   Input:
      %     opts: A struct with the following fields
      %       mtrcs: A cell array of the matrices or matrix specification structs. 
      %         Each matrix is used for one dimension, with the matrix for the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements.  -1 indicates maximum number of
      %         measurements: prod(dims) - bgn_ofst
      if nargin == 0
        return
      else
        obj.setSensingMatrixMD(opts);
      end
    end
  
    function set(obj, opts)
      %   Input:
      %     obj: This object
      %     opts: A struct with the following fields
      %       mtrcs: A cell array of the matrices or matrix specification structs. 
      %         Each matrix is used for one dimension, with the matrix for the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements.  -1 indicates maximum number of
      %         measurements: prod(dims) - bgn_ofst
      if nargin == 0
        return
      elseif nargin == 1
        opts = struct();
      end
      obj.setSensingMatrixMD(opts);
    end
    
    function n_no_clip = nNoClip(obj)
      % Return the number of no-clip elements in the output vector.
      n_no_clip = length(obj.indcsNoClip());
    end
    
    function indcs = indcsNoClip(obj)
      % Return a the indices of the no-clip elements as an ascending (column)
      % list
      if ~obj.is_transposed
        if isequal(obj.indcs_no_clip, -1)
          [~,~,obj.indcs_no_clip] = ...
            intersect(obj.mtrx{2}.indcsNoClip(), obj.linear_indx);
          obj.indcs_no_clip = sort(obj.indcs_no_clip);
        end
        indcs = obj.indcs_no_clip;
      else
        if isequal(obj.indcs_no_clip_trnsp, -1)
          obj.mtrx{2}.transpose();
          obj.indcs_no_clip_trnsp = obj.mtrx{2}.indcsNoClip();
          obj.mtrx{2}.transpose();
        end
        indcs = obj.indcs_no_clip_trnsp;
      end
    end
    
    function out = sortNoClip(obj, y)
      out = obj.sortNoClip@SensingMatrix(y);
    end
    function out = unsortNoClip(obj, y)
      out = obj.unsortNoClip@SensingMatrix(y);
    end
  end
  
  methods (Access = protected)
    function setSensingMatrixMD(obj, opts)
      %     opts: A struct with the following fields
      %       mtrcs: A cell array of the matrices or matrix specification structs. 
      %         Each matrix is used for one dimension, with the matrix for the
      %         first dimension first ("first dimension" is the one along which
      %         indices change fastest).
      %       bgn_ofst: (optional) The first measurements is bgn_ofst+1.
      %         Default: 0
      %       n_rows: Number of measurements. -1 indicates maximum number of
      %         measurements: prod(dims) - bgn_ofst
      
      mtrcs = opts.mtrcs;
      obj.dims = zeros(1, numel(mtrcs));
      for k=1:numel(mtrcs)
        if isstruct(mtrcs{k})
          mtrcs{k} = SensingMatrix.construct(mtrcs{k}.type, mtrcs{k}.args);
        end
        obj.dims(k) = mtrcs{k}.nRows();
      end
      nr1 = prod(obj.dims(:));
      
      if ~isfield(opts, 'bgn_ofst')
        opts.bgn_ofst = 0;
      end
      
      if ~isfield(opts, 'n_rows')
        error('n_rows not defined');
      elseif opts.n_rows == -1
        opts.n_rows = nr1 - opts.bgn_ofst;
      end
      
      mtrcs = mtrcs(end:-1:1);  % reverse order
      if length(mtrcs) == 1
        obj.linear_indx = (opts.bgn_ofst+1:opts.bgn_ofst+opts.n_rows)';
        obj.sbs_indx = obj.linear_indx;
        slct_mtrx = SensingMatrixSelectRange(opts.bgn_ofst+1,opts.bgn_ofst+opts.n_rows, nr1);
        kron_mtrx = mtrcs{1};
      else
         % Convert indices into subscripts
        sb_indx = zeros(nr1, length(obj.dims));
        for d = 1:length(obj.dims)
          sb_indx(:,d) = kron(ones(prod(obj.dims(d+1:end)),1), ...
            kron((1:obj.dims(d))', ones(prod(obj.dims(1:d-1)),1)));
        end
        
        % Compute subscripts sums
        sum_indx = sum(sb_indx,2);
        
        % Direction: positive even sums, negative for odd sums
        dir_indx = 1-2*mod(sum_indx,2);
        
        % First sort by the sum
        [~,indx] = sort(sum_indx);
        
        % Then sort lexicographically on indices
        chg_indx = zeros(ceil(length(indx)/2),1);
        done = false;
        while ~done;
          done = true;
          for bgn=1:2
            tst_indx = (bgn:2:nr1-1);
            n_chg = 0;
            tst_indx = tst_indx(sum_indx(indx(tst_indx)) == ...
              sum_indx(indx(tst_indx+1)));
            if isempty(tst_indx)
              continue
            end
            for d = 1:length(obj.dims)
              c_indx = ...
                find(dir_indx(indx(tst_indx)).*...
                (sb_indx(indx(tst_indx),d) - sb_indx(indx(tst_indx+1),d))<0);
              if ~isempty(c_indx)
                lc_indx = length(c_indx);
                done = false;
                chg_indx(n_chg+1:n_chg+lc_indx) = tst_indx(c_indx);
                n_chg = n_chg + lc_indx;
              end
              % Equality requires checking next dimension
              c_indx = ...
                find(sb_indx(indx(tst_indx),d) == sb_indx(indx(tst_indx+1),d));
              if isempty(c_indx)
                break;
              end
              tst_indx = tst_indx(c_indx);
            end
            c_indx = chg_indx(1:n_chg);
            indx([c_indx;c_indx+1]) = indx([c_indx+1; c_indx]);
          end
        end
        obj.linear_indx = indx(opts.bgn_ofst+1:opts.bgn_ofst+opts.n_rows);
        obj.sbs_indx = sb_indx(obj.linear_indx,:);
        slct_mtrx = SensingMatrixSelect(obj.linear_indx, nr1);
        kron_mtrx = SensingMatrixKron.construct(mtrcs);
      end
      obj.setSensingMatrixCascade({slct_mtrx, kron_mtrx});
      
      % compute no clip
      indcs = kron_mtrx.indcsNoClip();
      obj.setIndcsNoClip(indcs,true);
      if ~isempty(indcs)
        ind = zeros(kron_mtrx.nRows(),1);
        ind(indcs) = 1;
        ind = slct_mtrx.multVec(ind);
        indcs = find(ind);
      end
      obj.setIndcsNoClip(indcs, false);
    end
  end
  
end

