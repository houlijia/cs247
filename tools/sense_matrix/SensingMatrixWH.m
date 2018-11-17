classdef SensingMatrixWH < SensingMatrixSqrRnd
  % SensingMatrixWH Implementes a sensing matrix based on random selection of
  % rows and columns from a Walsh Hadamard matrix of appropriate order
  %   Detailed explanation goes here
  
  properties(Access=private)
    wh_mtrx = [];
  end
  
  properties(Access=protected)
    indcs_no_clip = -1;
    indcs_no_clip_trnsp = -1;
  end
  
  methods
    function obj = SensingMatrixWH(varargin)
      % Constructor
      %   Input: Can be either a single struct with the following fields, or a
      %   list of arguments corresponding to these fields, in this order
      %     n_rows - number of rows.
      %     n_cols - number of columns
      %     rnd_seed - random number generation seed
      %     prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.
      %     wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
      %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
      %     order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %     rnd_type - type of random number generator. Can be
      %        string - RandStream type
      %        0 - default type
      %        1 - Use RndCStrm
      %        2 - Use RndCStrm and compare with RandStream
      %     nodc - If present and true do not include the DC measurement in 
      %         the measurements vector
      if nargin > 0
        obj.set(varargin{:})
      end
    end
    
    function set(obj, varargin)
      % Set Initialize
      %   Input: 
      %     obj - this object
      %     The rest of the arguments, if present, can be either a single struct 
      %     with the following fields, or a list of arguments corresponding to 
      %     these fields, in this order
      %       obj - this object
      %       n_rows - number of rows.
      %       n_cols - number of columns
      %       rnd_seed - random number generation seed
      %       wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
      %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
      %       order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %       prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.      %
      %       rnd_type - type of random number generator. Can be
      %         string - RandStream type
      %         0 - default type
      %         1 - Use RndCStrm
      %         2 - Use RndCStrm and compare with RandStream
      %       nodc - If present and true do not include the DC measurement in 
      %         the measurements vector
      if nargin > 1
        obj.setSensingMatrixWH(varargin{:});
      end
    end
    
    function md = whMode(obj)
      md = obj.wh_mtrx.whMode();
    end
    
    function md_name = whModeName(obj)
      md_name = obj.wh_mtrx.whModeName();
    end
    
    function y = multSqr(obj,x)
      y = obj.wh_mtrx.multSqr(x);
    end
    
    function y = multTrnspSqr(obj,x)
      y = obj.wh_mtrx.multTrnspSqr(x);
    end
    
    % Override the same function of the superclass in order to insure that
    % the first measurement is selected.  Make sure that the first
    % entry of the output is the DC, i.e. PL(1)=1
    function setOrder(obj, order, opts)
      [PL, PR] = obj.makePermutations(order, opts);
      obj.log2order = obj.toCPUIndex(nextpow2(order));
      obj.setPermutations(order, PL, PR');
    end
        
    % normalize a measurements vector, so that if the input vector
    % components are independet, identically distributed random
    % variables, the each element of y will have the same variance as
    % the input vector elements (if the matrix is transposed, the
    % operation should be changed accordingly).
    function y=normalizeMsrs(obj,y)
      if ~obj.isTransposed();
        y = y / sqrt(obj.nCols());
      else
        y = y / (obj.nCols()/sqrt(obj.nCols()));
      end
    end
    
    % undo the operation of normalizeMsrs
    function y = deNormalizeMsrs(obj,y)
      if ~obj.isTransposed()
        y = y * sqrt(obj.nCols());
      else
        y = y * (obj.nCols()/sqrt(obj.nCols()));
      end
    end
    
    function y = cmpExactNorm(obj)
      if obj.n_cols < obj.sqr_order
        y = obj.cmpExactNorm@SensingMatrix();
      elseif ~obj.isTransposed()
        if obj.nRows() > 0
          y = sqrt(double(obj.nCols()));
        else
          y = 0;
        end
      else
        if obj.nCols() > 0
          y = sqrt(double(obj.nRows()));
        else
          y = 0;
        end
      end
    end
  end
  
  methods(Access=protected)
    function setSensingMatrixWH(obj, n_rows, n_cols, ...
        rnd_seed, prmt, wh_md, sqr_order, rnd_type, nodc)
      % Set Initialize
      %   Input: 
      %     obj - this object
      %     The rest of the arguments, if present, can be either a single struct 
      %     with the following fields, or a list of arguments corresponding to 
      %     these fields, in this order
      %       n_rows - number of rows
      %       n_cols - number of columns
      %       rnd_seed - random number generation seed
      %       wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
      %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
      %       sqr_order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %       prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.      %
      %       rnd_type - type of random number generator. Can be
      %         string - RandStream type
      %         0 - default type
      %         1 - Use RndCStrm
      %         2 - Use RndCStrm and compare with RandStream
      %       nodc - If present and true do not include the DC measurement in 
      %         the measurements vector
      if nargin < 2
        return;
      elseif nargin == 2
        if isstruct(n_rows)
          opts = n_rows;
        else
          error('n_rows specified but not n_cols');
        end
      else
        % Converts arguments into struct opts
        opts = struct('n_rows', n_rows, 'n_cols', n_cols);
        if nargin >= 4
          if isstruct(rnd_seed)
            opts.rnd_seed = rnd_seed.seed;
            opts.rnd_type = rnd_seed.type;
          else
            opts.rnd_seed = rnd_seed;
          end
        end
        if nargin >= 5
          opts.prmt = prmt;
        end
        if nargin >= 6
          opts.wh_md = wh_md;
        end
        if nargin >= 7
          opts.sqr_order = sqr_order;
        end
        if nargin >= 8
          opts.rnd_type = rnd_type;
        end
        if nargin >= 9
          opts.nodc = nodc;
        end
      end
      
      % Set some default values
      if ~isfield(opts, 'prmt')
        opts.prmt = struct();
      end
      if ~isfield(opts.prmt, 'PL_mode')
        if isa(obj, 'SensingMatrixNrWH')
          opts.prmt.PL_mode = SensingMatrixSqr.SLCT_MODE_ZGZG;
        else
          opts.prmt.PL_mode = SensingMatrixSqr.SLCT_MODE_ARBT;
        end
      end
      if ~isfield(opts, 'wh_md')
        opts.wh_md = 'hadamard';
      end
      if ~isfield(opts, 'nodc')
        opts.nodc = false;
      end
      if ~isfield(opts,'sqr_order')
        opts.sqr_order = obj.defaultOrder(opts.n_rows, opts.n_cols, opts.prmt);
      end
      opts.order = opts.sqr_order;
      
      % Initialize base matrix
      obj.setSensingMatrixSqrRnd(opts)
      obj.wh_mtrx = SensingMatrixBasicWH(opts);
      
      if obj.n_cols == obj.sqr_order
        obj.setOrtho_row(true);
      end
      if obj.n_rows == obj.sqr_order
        obj.setOrtho_col(true);
      end
    end
    
    function order = defaultOrder(obj, n_rows, n_cols, prmt)
      log2ord = double(nextpow2(max(n_cols, n_rows))); % log2 rounded up
      if prmt.PL_mode == SensingMatrixWH.SLCT_MODE_ZGZG 
        % If we use SLCT_MODE_ZGZG make sure that log2ord is even
        log2ord = log2ord + mod(log2ord,2);
      end
      order = obj.toCPUIndex(pow2(log2ord));
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqrRnd();
      if ~isempty(obj.wh_mtrx)
        obj.wh_mtrx.setCastIndex();
      end
    end
    
    function PL = makeRowSlctPermutation(obj,order, opts)
      prmt = opts.prmt;
      if ~isfield(opts, 'nodc') || ~opts.nodc
        switch prmt.PL_mode
          case SensingMatrixConvolve.SLCT_MODE_NONE
            PL = 1:obj.n_rows;
          case SensingMatrixWH.SLCT_MODE_ARBT
            PL = [1, 1+obj.rnd_strm.randperm(order-1, obj.n_rows-1)]';
          case SensingMatrixWH.SLCT_MODE_ZGZG
            PL = index_zigzag(order, order);
            PL = PL(1:obj.n_rows);
          otherwise
            error('SensingMatrixWH:makeRowSlctPermutation',...
              'Unexpected row selection mode: %d', prmt.PL_mode);
        end
      elseif obj.n_rows >= order
        error('cannot have n_rows >= order with nodc');
      else
        switch prmt.PL_mode
          case SensingMatrixConvolve.SLCT_MODE_NONE
            PL = 2:obj.n_rows+1;
          case SensingMatrixWH.SLCT_MODE_ARBT
            PL = 1+obj.rnd_strm.randperm(order-1, obj.n_rows)';
          case SensingMatrixWH.SLCT_MODE_ZGZG
            PL = index_zigzag(order, order);
            PL = PL(2:obj.n_rows+1);
          otherwise
            error('SensingMatrixWH:makeRowSlctPermutation',...
              'Unexpected row selection mode: %d', prmt.PL_mode);
        end
      end
    end
  end
  
end

