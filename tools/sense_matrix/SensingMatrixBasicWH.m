classdef SensingMatrixBasicWH < SensingMatrix
  % SensingMatrixBasicWH is a square, WH matrix of size which a power of two.
  
  properties
    % Determines whether to use Matlab' built in fast WH functions or
    % the .mex files.
    use_matlab_WHtransform = false;
    
    log2order;
    sqr_order;

    % WH mode can be 1,2,3 for 'hadamard','dyadic','sequency'
    wh_mode ;
  end
  
  properties (Constant)
    wh_mode_names = {'hadamard','dyadic','sequency'};
  end
  
  properties(Access=protected)
    dc_indx = 1;
  end
  
  methods
    function obj = SensingMatrixBasicWH(opts)
      % Constructor
      %   Input:
      %     opts: A struct with the following fields (other fields are ignored)
      %       wh_md - (optional) Walsh Hadamard mode. Can be 1,2,3 or the 
      %         equivalent value strings 'hadamard','dyadic','sequency'. 
      %         Default: 'sequency'
      %       order - (required) order of the Walsh Hadamard matrix (power of 2).
      if nargin > 0
        obj.setSensingMatrixBasicWH(opts);
      end
    end
    
    function set(obj, opts)
      % Set Initialize
      %   Input:
      %     obj - this object
      %     opts: A struct with the following fields (other fields are ignored)
      %       wh_md - (optional) Walsh Hadamard mode. Can be 1,2,3 or the 
      %         equivalent value strings 'hadamard','dyadic','sequency'. 
      %         Default: 'sequency'
      %       order - (required) order of the Walsh Hadamard matrix (power of 2).
      if nargin > 0
        obj.setSensingMatrixBasicWH(opts);
      end
    end
    
    function md = whMode(obj)
      md = obj.wh_mode;
    end
    
    function md_name = whModeName(obj)
      md_name = obj.wh_mode_names(obj.wh_mode);
    end
    
    function setTransform(obj, use_matlab)
      obj.use_matlab_WHtransform = obj.toLogical(use_matlab);
    end
    
    function y = doMultVec(obj,x)
      y = obj.multSqr(x(:));
    end
    
    function y = doMultMat(obj,x)
      y = obj.multSqr(x);
    end
    
    function y = doMultTrnspVec(obj,x)
      y = obj.multTrnspSqr(x(:));
    end
    
    function y = doMultTrnspMat(obj,x)
      y = obj.multTrnspSqr(x);
    end
    
    function y = cmpExactNorm (obj)
      y = sqrt(obj.toCPUFloat(obj.sqr_order));
    end
    
    function y = multSqr(obj,x)
      if obj.use_matlab_WHtransform
        y = ifwht(x, obj.n_cols, obj.wh_mode_names(obj.wh_mode));
      else
        y = wht_mex(x);
        if obj.wh_mode > 1
          y(obj.getModeReorderSeq(obj.log2order, obj.wh_mode),:) = y;
        end
      end      
      %       rw = (1:size(x,2));
      %
      %       yc = arrayfun(@(k) {obj.multSqr(x(:,k))}, rw);
      %       y_ref = horzcat(yc{:});
      %       if norm(double(y(:))-double(y_ref(:)))/(norm(double(y_ref(:)))+1E-10) > 1E-6
      %         error('y_ref, y mismatch');
      %       end
    end
    
    function y = multTrnspSqr(obj,x)
      y = obj.multSqr(x);
    end
    
    function dc_val = getDC(obj,msrs)
      % Returns the sum of values of the measurement which contain DC value,
      % weighted by the ratio of the DC value to other components (in
      % terms of RMS), or 0 if there is no such measurement.
      %   Input:
      %     obj - this object
      %     msrs - the measurements vector
      dc_val = msrs(obj.dc_indx);
    end
  end
  
  methods(Access=protected)
    function setSensingMatrixBasicWH(obj, opts)
      % Set Initialize
      %   Input:
      %     obj - this object
      %     opts: A struct with the following fields (other fields are ignored)
      %       wh_md - (optional) Walsh Hadamard mode. Can be 1,2,3 or the 
      %         equivalent value strings 'hadamard','dyadic','sequency'. 
      %         Default: 'sequency'
      %       order - (required) order of the Walsh Hadamard matrix (power of 2).
      if nargin < 2
        return;
      end
      
      log2ord = nextpow2(opts.order);
      ord = pow2(double(log2ord));
      if double(opts.order) ~= ord;
        error('Order is %d, not a power of two', opts.order);
      end
      obj.setSensingMatrix(opts.order, opts.order);

      obj.log2order = obj.toCPUIndex(log2ord);
      obj.sqr_order = obj.toCPUIndex(ord);

      if ~isfield(opts, 'wh_md')
        opts.wh_md = 'sequency';
      end
      
      switch opts.wh_md
        case {1, 2, 3}
          obj.wh_mode = opts.wh_md;
        case 'hadamard'
          obj.wh_mode = 1;
        case 'dyadic'
          obj.wh_mode = 2;
        case 'sequency'
          obj.wh_mode = 3;
        otherwise
          error('Illegal Walsh-Hadamard mode');
      end
      
      obj.setOrtho(true);
      obj.setIndcsNoClip(1, false);
      obj.setIndcsNoClip(1, true);
    end
    
    function setCastIndex(obj)
      if ~isempty(obj.toLogical)
        obj.use_matlab_WHtransform = obj.toLogical(obj.use_matlab_WHtransform);
      end
      obj.log2order = obj.toCPUIndex(obj.log2order);
    end    
    
  end
  
  methods(Static)
    function y = do_ifWHtrans(x, log2ordr, mode)
      y = ifWHtrans(x);
      if mode < 3
        % The mex function performs the transform in sequency
        % order.  We need to convert
        y(SensingMatrixBasicWH.getReorderSeq(log2ordr, mode)) = y;
        
      end      
    end
    
    function seq = getModeReorderSeq(log2order, mode)
      persistent seqs;
      
      mode = mode-1;
      
      if isempty(seqs)
        seqs = cell(2,32);
      end
      
      if isempty(seqs{mode, log2order})
        order = pow2(double(log2order));
        br = bitrevorder(1:order);
        % br is its own inverse, so no explict inversion is necessary
        if mode == 1
          % Convert to dyadic
          seqs{mode,log2order} = br;
        else
          % Convert to sequency
          gr = SensingMatrixBasicWH.grayOrder(order);
          % inversion:
          inv_gr = 1:length(gr);
          inv_gr(gr) = inv_gr;
          seqs{mode,log2order} = inv_gr(br);
        end
      end
      seq = seqs{mode,log2order};
    end
    
    function seq = getReorderSeq(log2order, mode)
      persistent seqs;
      
      if isempty(seqs)
        seqs = cell(2,32);
      end
      
      if isempty(seqs{mode,log2order})
        order = pow2(double(log2order));
        switch(mode)
          case 1
            br = bitrevorder(1:order);
            gr = SensingMatrixBasicWH.grayOrder(order);
            seqs{mode,log2order} = br(gr);
          case 2
            seqs{mode,log2order} = SensingMatrixBasicWH.grayOrder(order);
        end
      end
      seq = seqs{mode,log2order};
    end
    
    function seq = grayOrder(order)
      seq = 0:order-1;
      seq = bitxor(seq, bitshift(seq,-1));
      seq = (seq+1)';
    end
    
  end
  
end