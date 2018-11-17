classdef SensingMatrixDCT  < SensingMatrixSqrRnd
  % SensingMatrixDFT Implementes a sensing matrix based on random selection of
  % rows and columns from a DFT transform of appropriate order
  
  properties
    % Determines whether to use Matlab' built in fast WH functions or
    % the .mex files.
    log2order;
  end
  
  
  methods
    function obj = SensingMatrixDCT(varargin)
      % Constructor
      %   Input:
      %     num_rows - number of rows, or a struct that has a field
      %         'N_msrs', which specifies the number of rows.
      %     num_columns - number of columns
      %     rnd_seed - random number generation seed
      %     prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.
      %     order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %     rnd_type - type of random number generator. Can be
      %        string - RandStream type
      %        0 - default type
      %        1 - Use RndCStrm
      %        2 - Use RndCStrm and compare with RandStream
      %     nodc - Do not include the DC measurement in the measurements vector
      obj.set(varargin{:})
    end
    
    function set(obj, varargin)
      % Set Initialize
      %   Input:
      %     obj - this object
      %     num_rows - number of rows, or a struct that has a field
      %         'N_msrs', which specifies the number of rows.
      %     num_columns - number of columns
      %     rnd_seed - random number generation seed
      %     prmt - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.
      %     order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %     rnd_type - type of random number generator. Can be
      %        string - RandStream type
      %        0 - default type
      %        1 - Use RndCStrm
      %        2 - Use RndCStrm and compare with RandStream
      %     nodc - Do not include the DC measurement in the measurements vector
      varargin = parseInitArgs(varargin, {'n_rows', 'n_cols', ...
        'rnd_seed', 'prmt', 'order', 'rnd_type'});
      obj.setSensingMatrixDCT(varargin{:});
      obj.setIndcsNoClip([], false);
    end
    
    % Override the same function of the superclass in order to insure that
    % the first measurement is selected.  Make sure that the first
    % entry of the output is the DC, i.e. PL(1)=1
    function setOrder(obj, order, opts)
      [PL, PR] = obj.makePermutations(order,opts);
      obj.log2order = obj.toCPUIndex(nextpow2(order));
      obj.setPermutations(order, PL, PR');
    end
    
    function y=multSqr(~,x)
      y = dct(x);
    end
    
    function y = multTrnspSqr(~,x)
      y = idct(x);
    end
    
    function y=multSqrMat(~,x)
      y = dct(x);
    end
    
    function y = multTrnspSqrMat(~,x)
      y = idct(x);
    end
    % Returns the sum of values of the measurement which contain DC value,
    % weighted by the ratio of the DC value to other components (in
    % terms of RMS), or 0 if there is no such measurement.
    %   Input:
    %     obj - this object
    %     msrs - the measurements vector
    function dc_val = getDC(~,msrs)
      dc_val = msrs(1);
    end
    
    function y = cmpExactNorm(obj)
      if obj.n_cols < obj.sqr_order
        y = obj.cmpExactNorm@SensingMatrix();
      else
        y = 1;
      end
    end
  end
  
  methods(Access=protected)
    function setSensingMatrixDCT(obj,num_rows, num_columns, ...
        rnd_seed, prmt_info, order, rnd_type)
      %   Input:
      %     obj - this object
      %     num_rows - number of rows, or a struct that has a field
      %         'N_msrs', which specifies the number of rows.
      %     num_columns - number of columns
      %     rnd_seed - random number generation seed
      %     prmt_info - a struct containing parameters for premtutation and selection
      %              the only relevant field is PL_mode.      %
      %     order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %     rnd_type - type of random number generator. Can be
      %        string - RandStream type
      %        0 - default type
      %        1 - Use RndCStrm
      %        2 - Use RndCStrm and compare with RandStream
      %     nodc - Do not include the DC measurement in the measurements vector
      if nargin < 2
        return;
      elseif nargin < 3
        error('n_rows specified but not n_cols');
      else
        if isstruct(num_rows)
          num_rows = num_rows.N_msrs;
        end
        if nargin < 4
          rnd_seed = SensingMatrixRnd.default_seed;
        end
        switch nargin
          case {3,4}
            smr_args = { num_rows, num_columns, rnd_seed};
          case 5
            smr_args = { num_rows, num_columns, rnd_seed, prmt_info};
          case 6
            smr_args = { num_rows, num_columns, rnd_seed, ...
              prmt_info, order};
          case 7
            smr_args = { num_rows, num_columns, rnd_seed, ...
              prmt_info, order, rnd_type};
          otherwise
            error('Unexpected number of arguments');
        end
      end
      obj.setSensingMatrixSqrRnd(smr_args{:});
      if nargin >= 3 && num_rows <= num_columns
        obj.setOrtho_row(true);
      end
    end
    
    function order = defaultOrder(obj, num_rows, num_columns, ~)
      order = obj.toCPUIndex(pow2(double(nextpow2(max(num_columns, num_rows)))));
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixSqrRnd();
      obj.log2order = obj.toCPUIndex(obj.log2order);
    end
    
  end
end

