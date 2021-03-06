classdef SensingMatrixNrDFT < SensingMatrixSqrNrRnd & SensingMatrixDFT
  % SensingMatrixNrDFT is the same as SensingMatrixDFT, but
  % pre-randomization is not done in order to allow motion detection
  
  properties
  end
  
  methods
    function obj = SensingMatrixNrDFT(varargin)
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
      %              the only relevant field is PL_mode.      %
      %     order - order of the Walsh Hadamard matrix (power of 2).
      %            if not specificed the lowest possible value is used.
      %     rnd_type - type of random number generator. Can be
      %        string - RandStream type
      %        0 - default type
      %        1 - Use RndCStrm
      %        2 - Use RndCStrm and compare with RandStream
      %     nodc - Do not include the DC measurement in the measurements vector
      varargin = parseInitArgs(varargin, {'n_rows', 'n_cols',...
        'rnd_seed', 'prmt', 'order', 'rnd_type'});
      obj.setSensingMatrixNrDFT(varargin{:});
    end
    
    % Get an array of measurements which correspond to specific offsets.
    %   Input
    %   Input:
    %     obj: This object
    %     ofsts: a vector of offsets of length lofst.
    %     inp_list: A list of input measurement numbers to use
    %               ofsts_list - a column vector of indices of columns of
    %                           ofst_msrs (see below)
    %               nghbr_list - an array of indices of columns of
    %                            ofst_msrs. The number of rows is the
    %                            same as length(params.ofsts_list
    %   Output
    %     indcs: An array with rows of lengths lofst. Each row
    %           contains indices in inp_list such that if i is
    %           the n-th index in the row and j is the k-th index, then
    %           obj.PL(inp_list(j))-obj.PL(inp_list(i)) =
    %              ofsts(k) - ofst(n)  mod(obj.sqr_order)
    %
    %  Note: If params is present and has the fields ofsts_list
    %        and nghbr_list, then after computing ofst_msrs it is
    %        modified by a call
    %    ofst_msrs = obj.getEdgeMsrmnts(ofst_msrs,
    %                                    params.ofsts_list,
    %                                    params.nghbr_list);
    %        or something functionally equivalent.
    function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts, msrs,...
        inp_list, params)
      if nargin < 5
        params = struct();
      end
      if ~isfield(params, 'nrm_exp')
        params.nrm_exp = 1;
      end
      if ~isnumeric(inp_list)
        inp_list = 1:obj.nRows();
      end
      
      if ~isempty(obj.zeroed_rows)
        [~,zr,~] = intersect(inp_list, obj.zeroed_rows);
        inp_list(zr) = [];
      end
      
      [tlist, cmsrs, nrl] = obj.sortMsrsList(msrs, inp_list);
      
      % Multiply entries which are not conjugates of themselves to
      % account for them appearing twice in the full DFT
      cmsrs(nrl+1:end) = cmsrs(nrl+1:end) * (2 ^ (1./params.nrm_exp));
      
      wgts = obj.getShiftWgts(obj.log2order);
      if isa(tlist, 'gpuArray')
        wgt_ind = mod(obj.toFloat(tlist)*obj.toFloat(ofsts(:)'), ...
          obj.toCPUFloat(obj.sqr_order))+1;
      else
        wgt_ind = mod(tlist*ofsts(:)', obj.sqr_order)+1;
      end
      ofst_msrs_c = (cmsrs * obj.ones(1,length(ofsts))) .* wgts(wgt_ind);
      
      if isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
        ofst_msrs_c  = obj.getEdgeMsrmnts(ofst_msrs_c, ...
          params.ofsts_list, params.nghbr_list);
      end
      
      ofst_msrs = obj.convertToReal(ofst_msrs_c,nrl);
    end
        
  end
  
  methods (Access = protected)
    function setSensingMatrixNrDFT(obj,num_rows, num_columns, ...
        rnd_seed, prmt_info, order, rnd_type)
      switch nargin
        case 1; args = {};
        case 3; args = {num_rows, num_columns};
        case 4; args = {num_rows, num_columns, rnd_seed};
        case 5; args = {num_rows, num_columns, rnd_seed, prmt_info};
        case 6; args = {num_rows, num_columns, rnd_seed, prmt_info, order};
        case 7; args = {num_rows, num_columns, rnd_seed, prmt_info, order, rnd_type};
      end
      obj.setSensingMatrixDFT(args{:});
    end
    
    function [PL, PR] = makePermutations(obj, order, opts)
      PL = obj.makeRowSlctPermutation(order, opts);
      PR = [];
    end
    
    function setCastIndex(obj)
      obj.setCastIndex@SensingMatrixDFT();
    end
    
  end
end

