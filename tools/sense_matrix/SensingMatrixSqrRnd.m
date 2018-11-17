classdef SensingMatrixSqrRnd < SensingMatrixSqr & SensingMatrixRnd
  %SensingMatrixSqrRnd defines a SensingMatrixSqr where the permutations
  %are defined randomly.
  
  properties (Constant)
    % These fields should normally be false and set only for debugging.
    % If true, they override the randomization and cause generation of
    % unit permutations.
    unit_permut_R = false;
  end
  
  properties (SetAccess=protected)
    nodc = false;
  end
  
  methods
    function obj = SensingMatrixSqrRnd(varargin)
      %Constructor
      %Input arguments (optional):
      %   Input: either a single struct 
      %     with the following fields, or a list of arguments corresponding to 
      %     these fields, in this order
      %  num_rows - number of rows
      %  num_columns - number of cloumns (must be present if num_rows is present)
      %  rnd_seed - seed of random number generator
      %  prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.
      %  sqr_order - order of square matrix
      %  rnd_type - type of random number generator
      %  nodc - If present and true do not include the DC measurement in 
      %    the measurements vector
      obj.setSensingMatrixSqrRnd(varargin{:});
    end
    
    function set(obj, varargin)
      %Set parameters for the object
      %   Input: 
      %     obj - this object
      %     The rest of the arguments, if present, can be either a single struct 
      %     with the following fields, or a list of arguments corresponding to 
      %     these fields, in this order
      %  num_rows - number of rows
      %  num_columns - number of cloumns (must be present if num_rows is present)
      %  rnd_seed - seed of random number generator
      %  prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.
      %  sqr_order - order of square matrix
      %  rnd_type - type of random number generator
      %  nodc - If present and true do not include the DC measurement in 
      %    the measurements vector
      obj.setSensingMatrixSqrRnd(varargin{:});
    end
    
    function setOrder(obj, sqr_order, opts)
      [PL, PR] = obj.makePermutations(sqr_order, opts);
      obj.setPermutations(sqr_order, PL', PR');
    end
    
  end
  
  methods(Access=protected)
    function setSensingMatrixSqrRnd(obj, num_rows, num_columns, ...
        rnd_seed, prmt, sqr_order, rnd_type, nodc)
      %   Input: 
      %     obj - this object
      %     The rest of the arguments, if present, can be either a single struct 
      %     with the following fields, or a list of arguments corresponding to 
      %     these fields, in this order
      %  num_rows - number of rows
      %  num_columns - number of cloumns (must be present if num_rows is present)
      %  rnd_seed - seed of random number generator
      %  prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.      %  
      %  sqr_order - order of square matrix
      %  rnd_type - type of random number generator
      %  nodc - If present and true do not include the DC measurement in 
      %    the measurements vector
      if nargin < 2
        return
      elseif nargin == 2
        opts = num_rows;
        if ~isfield(opts,'prmt')
          opts.prmt = struct();
        end
      else
        opts = struct('n_cols', num_columns, 'n_rows', num_rows);
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
          opts.sqr_order = sqr_order;
        end
        if nargin >= 7
          opts.rnd_type = rnd_type;
        end
        if nargin >= 8
          opts.nodc = nodc;
        end
      end
      
      % Set defaults
      if ~isfield(opts, 'prmt')
        opts.prmt = struct();
      end
      if isfield(opts.prmt, 'N_msrs') && ~isempty(opts.prmt.N_msrs)
        if isfield(opts, 'n_rows') && ~isempty(opts.n_rows)
          if ~isequal(opts.prmt.N_msrs, opts.n_rows)
            error('opts.n_rows and opts.prmt.N_msrs are different');
          end
        else
          opts.n_rows = opts.prmt.N_msrs;
        end
      elseif isfield(opts, 'n_rows') && ~isempty(opts.n_rows)
        opts.prmt.N_msrs = opts.n_rows;
      end
      
      if ~isfield(opts,'sqr_order')
        opts.sqr_order = obj.defaultOrder(opts.n_rows, opts.n_cols, opts.prmt);
      end
      
      if ~isfield(opts, 'nodc')
        opts.nodc = false;
      end
      
      obj.setSensingMatrixRnd(opts);
      
      opts.mltSqr = @obj.multSqr;
      opts.mltTrnspSqr = @obj.multTrnspSqr;
      obj.nodc = opts.nodc;
      [opts.PL, opts.PR] = obj.makePermutations(opts.sqr_order, opts);
      
      obj.setSensingMatrixSqr(opts);
      
      obj.setIndcsNoClip(1, false);
      obj.setIndcsNoClip([], true);
    end
    
    function [PL, PR] = makePermutations(obj, sqr_order, opts)
      PL = obj.makeRowSlctPermutation(sqr_order, opts);
      if obj.unit_permut_R
        PR = 1:sqr_order;
      else
        PR = obj.makeRandomizerPermutation(sqr_order);
      end
    end
    
    function PL = makeRowSlctPermutation(obj,sqr_order,opts)
      if ~isfield(opts, 'nodc') || ~opts.nodc
        PL = [1, 1+obj.rnd_strm.randperm(sqr_order-1, obj.n_rows-1)]';
      else
        PL = 1+obj.rnd_strm.randperm(sqr_order-1, obj.n_rows)';
      end
    end
    
    function PR = makeRandomizerPermutation(obj, sqr_order)
      PR = obj.rnd_strm.randperm(sqr_order);
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqr();
    end
    
  end
  
end

