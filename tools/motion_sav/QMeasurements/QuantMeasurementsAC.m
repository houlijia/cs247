classdef QuantMeasurementsAC < QuantMeasurements  & CodeElement
    %QuantMeasurementsAC - implementation of QuantMeasurements where values
    %are coded using arithmetic coding.
    

    properties(Constant)
        PROB_MLTPLR = 1e6;
        min_stdv = 1e-10;   % Minimum standard deviation
        
        force_use_gaus = false;
    end
    
    methods
        function obj = QuantMeasurementsAC(nnclp,mn,sd,bn,sv)
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
        
        function len = encode(obj, code_dest, info)
            max_bin = (obj.n_bins+1)/2;
            if info.quantizer.q_ampl_mltplr / max_bin <...
                    info.enc_opts.lossless_coder_AC_gaus_thrsh
                use_gaus = 1;
            else
                use_gaus = 0;
            end
%             fprintf('......... use_gaus=%d. ampl/n_bins=%f/%d = %f. Thrs=%f\n',...
%                 use_gaus, info.quantizer.q_ampl_mltplr, (obj.n_bins+1),...
%                 info.quantizer.q_ampl_mltplr/(obj.n_bins+1), ...
%                 info.enc_opts.lossless_coder_AC_gaus_thrsh);
            len_a = length(obj.bin_numbers) - obj.n_no_clip;
            len_s = length(obj.saved);
                        
            if len_a;
                % Compute atithmetic encoding: Get probabilities of bins, conver
                % them to integer counts and run arithmetic encoder on clipped
                % bin_numbers
                [freq, freq_to_code] = obj.calcFreq(info.quantizer, use_gaus);
                arith_bins = arithenco(obj.bin_numbers(obj.n_no_clip+1:end), freq);
            else
                freq_to_code = [];
                arith_bins = [];
            end
            
            % Write lengths (unsigned integers)
            len = code_dest.writeUInt(...
                [obj.n_bins, obj.n_no_clip, len_a, len_s, use_gaus]);
            if ischar(len);  return; end
            
            % Write floating point parametrs
            n = code_dest.writeNumber([obj.mean_msr, obj.stdv_msr]);
            if ischar(n); len = n; return; end
            len = len + n;

            % Write information defining histogram of labels
            n = obj.encodeHist(code_dest, freq_to_code);
            if ischar(n); len = n; return; end
            len = len + n;
            
            % Write unclipped bins (signed integers)            
            n = code_dest.writeSInt(obj.bin_numbers(1:obj.n_no_clip));
            if ischar(n); len = n; return; end
            len = len + n;

            % Write arithmetically encoded bins
            n = code_dest.writeBits(arith_bins);
            if ischar(n); len = n; return; end
            len = len + n;
            
            % Write saved bins (signed integers)
            n = code_dest.writeSInt(obj.saved);
            if ischar(n); len = n; return; end
            len = len + n;
        end
        
        function len = decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            % Read lengths (unsigned integers)
            if QuantMeasurementsAC.force_use_gaus
                [vals, nread] = code_src.readUInt(cnt, [1,4]);
                if ischar(vals) || (length(vals)==1 && vals == -1)
                    len = vals;
                    return
                end
                use_gaus = 1;
            else
                [vals, nread] = code_src.readUInt(cnt, [1,5]);
                if ischar(vals) || (length(vals)==1 && vals == -1)
                    len = vals;
                    return
                end
                use_gaus = vals(5);
            end
            nbn = double(vals(1));
            nnclp = vals(2);
            len_a = vals(3);
            len_s = vals(4);
            cnt = cnt - nread;
            len = nread;
            
            % Read mean and standard deviation
            [valstat, nread] = code_src.readNumber(cnt, [1,2]);
            if ischar(valstat)
                % error occurred in reading
                if isempty(valstat)
                    len = 'Unexpected end of Data';
                else
                    len = valstat;
                end
                return;
            end
            len = len+nread;
            
            % read histogram information
            [freq, nread] = obj.decodeHist(code_src, cnt, info, use_gaus, ...
                valstat(2), nbn, len_a);
            if ischar(freq)
                len = valstat;
                return
            end
            cnt = cnt - nread;
            len = len + nread;
            
            % Read unclipped bins (signed integers)
            [vals1, nread] = code_src.readSInt(cnt, [nnclp, 1]);
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
            
            % Read arithmetically encoded clipped bins (byte sequence)
            [arith_bins, nread] = code_src.readBits(cnt);
            if ischar(arith_bins); len = arith_bins; return; end
            cnt = cnt - nread;
            len = len + nread;
            
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
            
            % struct obj.arith_info is used by obj.get.bin_numbers
            obj.arith_info = struct('vals1',vals1, 'len_a', len_a,...
                'arith_bins',arith_bins, 'freq', freq);
            
            obj.n_bins = double(nbn);
            obj.n_no_clip = nnclp;
            obj.mean_msr = valstat(1);
            obj.stdv_msr = valstat(2);
            if isempty(vals)
                obj.saved = [];
            else
                obj.saved = double(vals);
            end
        end
        
    end
    
    methods (Access=protected)
        % calcFreq - calculate frequences for artihmetic coding
        function [freq, freq_to_code] = calcFreq(obj, qntzr, use_gaus)
            if use_gaus
                freq = obj.calcFreqGaus(qntzr, obj.stdv_msr, obj.n_bins);
                freq_to_code = [];
            else
                freq_to_code = ...
                    histc(double(obj.bin_numbers(obj.n_no_clip+1:end)),...
                    1:(obj.n_bins+1));
                freq = obj.calcFreqHist(freq_to_code);
            end
        end
        
        function freq = calcFreqGaus(obj, qntzr, stdv, nbn)
            stdv = max(obj.min_stdv, stdv);
            [prob, prob_out] = qntzr.binGaussProb(stdv, nbn);
            freq = round(([prob, prob_out] * obj.PROB_MLTPLR) + 1);
        end
        
        function freq = calcFreqHist(obj, freq_to_code)
            freq = double(freq_to_code) * obj.PROB_MLTPLR;
            freq(freq_to_code==0)=1;
        end            
        
        function n = encodeHist(~, code_dst, freq)
            n = code_dst.writeUInt(freq);
        end

        function [freq, nread] = decodeHist(obj, code_src, cnt, info, ...
                use_gaus, stdv, nbn, len_a)
            if ~len_a
                freq = [];
                nread = 0;
                return
            end
            
            if use_gaus
                freq = obj.calcFreqGaus(info.quantizer, stdv, nbn);
                nread = 0;
            else
                [freq, nread] = code_src.readUInt(cnt, [(nbn+1),1]);
                if ischar(freq)
                    return
                elseif length(freq)==1 && freq==-1
                    freq = 'unexpected end of data';
                    return
                end
                
                freq = obj.calcFreqHist(freq);
            end
        end
    end
end

