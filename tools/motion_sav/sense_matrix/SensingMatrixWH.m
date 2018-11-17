classdef SensingMatrixWH < SensingMatrixSqrRnd
    % SensingMatrixWH Implementes a sensing matrix based on random selection of
    % rows and columns from a Walsh Hadamard matrix of appropriate order
    %   Detailed explanation goes here
    
    properties
        % Determines whether to use Matlab' built in fast WH functions or
        % the .mex files.
        use_matlab_WHtransform = false;  
        
        log2order;
    end
    
    properties (Constant)
        % WH mode can be 1,2,3 for 'hadamard','dyadic','sequency'
        wh_mode = 1;
        wh_mode_names = {'hadamard','dyadic','sequency'};
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
        function obj = SensingMatrixWH(varargin)
            obj.set(varargin{:})
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
            obj.setSensingMatrixWH(varargin{:});
        end
        
        function n_no_clip=nNoClip(~)
            n_no_clip = 1;
        end
        
        % Override the same function of the superclass in order to insure that
        % the first measurement is selected.  Make sure that the first
        % entry of the output is the DC, i.e. IPL(1)=1
        function setOrder(obj, order, dummy)
            [PL, PR] = obj.makePermutations(order, dummy);
            obj.log2order = nextpow2(order);
            obj.setPermutations(order, PL, PR');
        end
                
        function setTransform(obj, use_matlab)
            obj.use_matlab_WHtransform = use_matlab;
        end
        
        function y=multSqr(obj,x)
            if ~isa(x, 'double')
                x = double(x);
            end
            % Use inverse WH transform to avoid scaling
            if obj.use_matlab_WHtransform 
                y = ifwht(x, obj.sqr_order, obj.wh_mode_names{obj.wh_mode});
            else
                y = obj.do_ifWHtrans(x, obj.log2order, obj.wh_mode);
                % Checking
%                 err = y - ifwht(x, obj.sqr_order, obj.wh_mode_names{obj.wh_mode});
%                 err = max(abs(err(:)));
%                 if err
%                     fprintf('WH error: %g\n', err);
%                 end
            end
            
        end
        
        function y = multTrnspSqr(obj,x)
            % The matrix is symmetric and the forward WH transform 
            % is the same matrix as the inverse transform except for scaling.
            % Here we use the forward matrix in order to get the scaling.
            if obj.use_matlab_WHtransform 
                y = fwht(x, obj.sqr_order, obj.wh_mode_names{obj.wh_mode});
            else
                y = fWHtrans(x);
                if obj.wh_mode < 3
                    % The mex function performs the transform in sequency
                    % order.  We need to convert
                    y(obj.getReorderSeq(obj.log2order, obj.wh_mode)) = y;
                end
                % Checking
%                 err = y - fwht(x, obj.sqr_order, obj.wh_mode_names{obj.wh_mode});
%                 err = max(abs(err(:)));
%                 if err
%                     fprintf('WH error: %g\n', err);
%                 end
            end
        end
        
        function y = trnspScale(obj)
            y = 1./double(obj.sqr_order);
        end

        function y = normAtA(~)
            y = 1;
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          if ~obj.is_transposed
            y = y / sqrt(obj.nCols());
          else
            y = y / (obj.nCols()/sqrt(obj.nCols()));
          end
        end

        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          if ~obj.is_transposed
            y = y * sqrt(obj.nCols());
          else
            y = y * (obj.nCols()/sqrt(obj.nCols()));
          end
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
        function setSensingMatrixWH(obj,num_rows, num_columns, ...
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
    
    methods(Static)
        function y = do_ifWHtrans(x, log2ordr, mode)
            y = ifWHtrans(x);
            if mode < 3
                % The mex function performs the transform in sequency
                % order.  We need to convert
                y(SensingMatrixWH.getReorderSeq(log2ordr, mode)) = y;
                
            end
        end
        
        function seq = getReorderSeq(log2order, mode)
            persistent seqs;
            
            if isempty(seqs)
                seqs = cell(2,32);
            end
            
            if isempty(seqs{mode,log2order})
                order = pow2(log2order);
                switch(mode)
                    case 1
                        br = bitrevorder(1:order);
                        gr = SensingMatrixWH.grayOrder(order);
                        seqs{mode,log2order} = br(gr);
                    case 2
                        seqs{mode,log2order} = SensingMatrixWH.grayOrder(order);
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

