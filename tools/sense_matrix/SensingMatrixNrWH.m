classdef SensingMatrixNrWH < SensingMatrixSqrNrRnd & SensingMatrixWH
    %SensingMatrixNrWH Same as SensingMatrixWH, except that the
    %input is not randomized
    
    properties
    end
    
    methods
      function obj = SensingMatrixNrWH(varargin)
        % Set Initialize
        %   Input:
        %     obj - this object
        %     n_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     n_cols - number of columns
        %     rnd_seed - random number generation seed
        %     prmt_info - a struct containing parameters for premtutation and selection
        %              the only relevant field is PL_mode.      %
        %     wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
        %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
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
        %     n_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     n_cols - number of columns
        %     rnd_seed - random number generation seed
        %     prmt_info - a struct containing parameters for premtutation and selection
        %              the only relevant field is PL_mode.      %
        %     wh_md - Walsh Hadamard mode. Can be 1,2,3 or the equivalent value
        %             strings 'hadamard','dyadic','sequency'. Default: 'hadamard'
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        %     nodc - Do not include the DC measurement in the measurements vector
        obj.set@SensingMatrixWH(varargin{:});
      end
      
      function md = whMode(obj)
        md = obj.whMode@SensingMatrixWH();
      end
    
      function md_name = whModeName(obj)
        md_name = obj.whModeName@SensingMatrixWH();
      end
    
      % Returns the sum of values of the measurement which contain DC value,
      % weighted by the ratio of the DC value to other components (in
      % terms of RMS), or 0 if there is no such measurement.
      %   Input:
      %     obj - this object
      %     msrs - the measurements vector
        function dc_val = getDC(obj,msrs)
            dc_val = obj.getDC@SensingMatrixWH(msrs);
        end
    end
    
    methods(Access=protected)
      function [PL, PR] = makePermutations(obj, order, opts)
        PL = obj.makeRowSlctPermutation(order, opts);
        PR = [];
      end
      
      function setCastIndex(obj)
        obj.setCastIndex@SensingMatrixWH();
      end
    end
    
end
