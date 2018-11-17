classdef QuantMeasurementsBasic < QuantMeasurements  & CodeElement
    %QuantMeasurementsBasic - implementation of QuantMeasurements where values
    %are coded using CodeDest number writing functions.
    
    properties (Constant)
        PROB_MLTPLR = 10^5;
    end
    
    methods
        function obj = QuantMeasurementsBasic(nnclp,mn,sd,bn,sv)
            switch (nargin)
                case 0
                    c_args = {};
                case 1
                    c_args = {nnclp};
                otherwise
                    c_args = {nnclp,mn,sd,bn,sv};
            end
            
            obj = obj@QuantMeasurements(c_args{:});
        end
        
        function len = encode(obj, code_dest, ~)
            len_b = length(obj.bin_numbers);
            len_s = length(obj.saved);
            
            % Write lengths (unsigned integers)
            len = code_dest.writeUInt([obj.n_bins, obj.n_no_clip, len_b, len_s]);
            if ischar(len);  return; end
            
            % Write mean and standard deviation (double)
            n = code_dest.writeNumber([obj.mean_msr, obj.stdv_msr]);
            if ischar(n); len = n; return; end
            len = len + n;
            
            % Write bin_numbers (signed integers)            
            n = code_dest.writeSInt(obj.bin_numbers - floor(obj.n_bins/2));
            if ischar(n); len = n; return; end
            len = len + n;
            
            % Write saved bins (signed integers)
            n = code_dest.writeSInt(obj.saved);
            if ischar(n); len = n; return; end
            len = len + n;
            
        end
        
        function len = decode(obj, code_src, ~, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            % Read lengths (unsigned integers)
            [vals, nread] = code_src.readUInt(cnt, [1,4]);
            if ischar(vals) || (length(vals)==1 && vals == -1)
                len = vals; 
                return
            end
            nbn   = vals(1);
            nnclp = vals(2);
            len_b = vals(3);
            len_s = vals(4);
            cnt = cnt - nread;
            len = nread;
            
            % read mean and standard deviation
            [valstat, nread] = code_src.readNumber(cnt, [1,2]);
            if ischar(valstat)
                if isempty(vlstat)
                    len = 'Unexpected end of Data';
                else
                    len = valstat; 
                end
                return
            end
            cnt = cnt - nread;
            len = len + nread;
            
            % Read unclipped bins (signed integers)
            [vals1, nread] = code_src.readSInt(cnt, [len_b, 1]);
            if ischar(vals1)
                if isempty(vals1)
                    len = 'Unexpected end of Data';
                else
                    len = vals1; 
                end
                return
            end
            cnt = cnt - nread;
            len = len + nread;
            vals1 = double(vals1);
            
            % Read saved bins (signed integers)
            [vals, nread] = code_src.readSInt(cnt, [len_s, 1]);
            if ischar(vals)
                if isempty(vals)
                    len = 'Unexpected end of Data';
                else
                    len = vals; 
                end
                return
            end
            len = len + nread;
            
            obj.n_bins = double(nbn);
            obj.n_no_clip = nnclp;
            obj.mean_msr = valstat(1);
            obj.stdv_msr = valstat(2);
            obj.bin_numbers = vals1 + floor(obj.n_bins/2);
            if isempty(vals)
                obj.saved = [];
            else
                obj.saved = double(vals);
            end
        end
    end    
end

