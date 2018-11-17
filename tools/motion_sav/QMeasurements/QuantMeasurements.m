classdef QuantMeasurements < handle
    %QuantMeasurements contains the output of quantization of measurements
    
    properties
        n_no_clip=0;      % The first n_no_clip entries in msrmnts are
    %                       divided by the quantization interval, but are
    %                       not clipped. The first n_no_clip measurements
    %                       also do not participate in mean and standard
    %                       deviation computation
        mean_msr = 0;     % mean of unquantized measurements
        stdv_msr = 0;     % standard deviation of unquantized measurements
        n_bins;           % number of bins in range (not including outlier bins)
        bin_numbers=[];   % an array of bin numbers. corresponding to the
    %                       measurements.
        saved=[];         % array of quantized values corresponding to clipped
    %                       values. These values are given as positive differnces
    %                       from the one before the last bin number of negative
    %                       differences from the one before the first bin numbers.
    
    end
    
    methods
        function obj = QuantMeasurements(nnclp,mn,sd,bn,sv)
            if nargin == 1 && isa(nnclp, 'QuantMeasurements');
                other = nnclp;
                obj.n_no_clip = other.n_no_clip;
                obj.mean_msr = other.mean_msr;
                obj.stdv_msr = other.stdv_msr;
                obj.bin_numbers = other.bin_numbers;
                obj.n_bins = other.n_bins;
                obj.saved = other.saved;
            elseif nargin > 0
                obj.n_no_clip = nnclp;
                obj.mean_msr = mn;
                obj.stdv_msr = sd;
                obj.bin_numbers = bn;
                obj.saved = sv;
            end
        end
        
        function [entrp, use_prob, use_entrp] = compEntropy(obj)
            vals = obj.bin_numbers(1+obj.n_no_clip:end);
            if isempty(vals)
                entrp = 0;
                use_prob = 1;
                use_entrp = entrp;
            else
                used_vals = vals(vals<=obj.n_bins);
                n_vals = length(vals);
                use_prob = length(used_vals)/n_vals;
                hst = hist(used_vals, max(used_vals)-min(used_vals)+1);
                hst = hst(hst ~= 0)/n_vals;  % discard zeros and calc frequencies
                entrp = - dot(hst, log(hst));
                use_entrp = entrp/use_prob;
            end
        end
        
        % This function is intended only for the arithmetic coding
        % subclass, QuantMeasurementsAC, otherwise it is a no operation. It
        % is used to defer the computation of arithmetic decoding of the
        % bin numbers until the first time that they are used.  This allows
        % arithmetic decoding, which is a heavy operation, to be performed
        % during parallel processing phase, rather than during decoding.
        function bnm = get.bin_numbers(obj)
            if ~isempty(obj.arith_info)
                vals1 = double(obj.arith_info.vals1);

                %perform arithmetic decoding
                if obj.arith_info.len_a > 0
                    vals2 = arithdeco(double(obj.arith_info.arith_bins), ...
                        obj.arith_info.freq, obj.arith_info.len_a);
                    if ~iscolumn(vals2)
                        vals2 = vals2';
                    end
                else
                    vals2 = zeros(0,1);
                end
                obj.bin_numbers = [vals1; vals2];

                obj.arith_info = [];
            end
            
            bnm = obj.bin_numbers;
        end
        
    end
    
    properties (Access=protected)
        arith_info = [];
    end
end

