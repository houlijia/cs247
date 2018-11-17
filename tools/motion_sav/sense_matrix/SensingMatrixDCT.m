classdef SensingMatrixDCT  < SensingMatrixSqrRnd
    % SensingMatrixDFT Implementes a sensing matrix based on random selection of
    % rows and columns from a DFT transform of appropriate order
   
    properties
        % Determines whether to use Matlab' built in fast WH functions or
        % the .mex files.
        log2order;
    end
    
    
    methods
        % Constructor
        %   Input:
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        function obj = SensingMatrixDCT(varargin)
            obj.setSensingMatrixDCT(varargin{:})
        end
        
        % Set Initialize
        %   Input:
        %     obj - this object
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixDCT(varargin{:});
        end
        
        function n_no_clip=nNoClip(~)
            n_no_clip = 1;
        end

        % Override the same function of the superclass in order to insure that
        % the first measurement is selected.  Make sure that the first
        % entry of the output is the DC, i.e. IPL(1)=1
        function setOrder(obj, order, ~)
            [PL, PR] = obj.makePermutations(order);
            obj.log2order = nextpow2(order);
            obj.setPermutations(order, PL, PR');
        end
                
        function y=multSqr(~,x)
            y = dct(x);
        end
        
        function y = multTrnspSqr(~,x)
            y = idct(x);
        end
            
        function y = trnspScale(~)
            y = 1;
        end

        function y = normAtA(~)
            y = 1;
        end
        
        function order = defaultOrder(~, num_rows, num_columns, ~)
            order = pow2(nextpow2(max(num_columns, num_rows)));
        end
        
        % Get the DC measurement
        function dc_val = getDC(~,msrs)
            dc_val = msrs(1);
        end
         
    end
    
    methods(Access=protected)
        % Set Initialize
        %   Input:
        %     obj - this object
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        function setSensingMatrixDCT(obj,num_rows, num_columns, ...
                rnd_seed, order, rnd_type)
            if nargin < 3
                smr_args = {};
            else
                if isstruct(num_rows)
                    num_rows = num_rows.N_msrs;
                end
                switch nargin
                    case 3
                        smr_args = { num_rows, num_columns };
                    case 4
                        smr_args = { num_rows, num_columns, rnd_seed};
                    case 5
                        smr_args = { num_rows, num_columns, rnd_seed, order};
                    case 6
                        % Inserting struct() for prmt_info
                        smr_args = { num_rows, num_columns, rnd_seed, ...
                            order, struct(), rnd_type};
                end
            end
            obj.setSensingMatrixSqrRnd(smr_args{:});
        end 
        
        function [PL, PR] = makePermutations(obj, order, ~)
            PL = [1, 1+obj.rnd_strm.randperm(order-1)]';
            PR = obj.rnd_strm.randperm(order);
        end
        
    end
end

