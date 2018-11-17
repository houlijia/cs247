classdef UniformQuantizer < CodeElement
    %UniformQuantizer Provides functions to quantize and unquantize sets of
    %measurements
    %   Detailed explanation goes here
    
properties (Constant=true)
    uniform_distr_stddev = 1/sqrt(12);
    
    % The optimal dequantizer does not quantize to the middle of the bin
    % but a little closer to the 0 because the guassian distribution is not
    % even throughout the bin.  If this is true, measurement will be
    % corrected accordingly.  A preliminary test showed that this caused a 
    % slight degradation, so this option is disabled.
    correct_gaus_unquant = false;
end

properties
    % Quantization step q_wdth is determined by q_wdth_mltplr.  If 
    % q_wdth_mltplr==0 there is no quantization (q_wdth=0, meaning that values 
    % are left as is, getting their integer value). If q_wdth_mltplr > 0 
    % the qunatization interval is 
    %    q_wdth = q_wdth_mltplr * q_wdth_unit
    % where q_wdth_unit is the quantization step for which the
    % variance of the quantization  error is the same as the variance of
    % the digitization error of the pixels as it appears in the
    % measurement (q_wdth_unit==0 means not set yet).  
    % Each pixel was derived by digitizing the analog video
    % samples into integer pixel values, hence the variance of the
    % digitization error is 1/12.  In the measurements, the average
    % digitization error is (1/m)*sum(a(i,j)^2, i=1:m, j=1:n)*(1/12) where a(i,j)
    % are the entries of the sensing matrix, which is of dimension m by n.
    % The variance of the quantization is q_wdth^2/12.  Therefore, 
    %   normalized_q_wdth = sqrt((1/m)*sum(a(i,j)^2, i=1:m, j=1:n))
    % In particular, if the entries of the sensing matrix have a magnitude
    % of 1, then normalized_q_wdth = sqrt(n).
    
    q_wdth_mltplr=0;
    q_wdth_unit=0;
    q_wdth=0;
    
    % The quantizer clips all values which exceeds a certain magnitude. This
    % magnitude is given by q_ampl_mltplr times the standard deviation of the
    % measurements.
    q_ampl_mltplr=0;
    
    % A flag indicating whether clipped values should be saved and returned as
    % a part of the quantization operation.
    save_clipped = false;
end

methods
    % Constructor
    % INPUT:
    %   q_wdth_mlt - value to be assigned to q_wdth_mltplr
    %   q_ampl_mlt - value to be assigned to q_ampl_mltplr
    %   sv_clp - if true the quantizer will save clipped values.
    %   q_unit  - (optional) interval unit.
    function obj = UniformQuantizer(q_w_mlt, q_ampl_mlt, sv_clp, q_unit)
        if nargin > 0
            obj.q_ampl_mltplr = q_ampl_mlt;
            obj.q_wdth_mltplr = q_w_mlt;
            obj.save_clipped = sv_clp;
            if nargin > 3
                obj.q_wdth = obj.calcIntvl(q_unit);
                obj.q_wdth_unit = q_unit;
            else
                 obj.q_wdth_unit = q_w_mlt * obj.uniform_distr_stddev;
            end
        end
    end
    
    function q_step = qStep(obj)
        if obj.q_wdth_mltplr
            q_step = obj.q_wdth;
        else
            q_step = 0;
        end
    end
    
    % quantize - quantize an array of measurements
    % Input:
    %   obj - the quantizer object
    %   msrmnts - the data array to quantize
    %   params  - is a struct of optional parameters. If the parameters are
    %             not present or equal [], a default values is used.  The possible
    %             fields are:
    %             n_no_clip - the number of measurements that should not be
    %                         clipped [default=0].  
    %             q_unit - a value of q_unit which overrides
    %                      obj.q_wdth_unit. 
    %             mean - a vaule to use as the mean of the measurements,
    %                    instead of a calculated value.
    %             stdv - a value to use as the standard deviation of the
    %                    measurements instead of a calculated value. A
    %                    value of -1 has a secial meaning.  Suppose the
    %                    measurements are x(1),...,x(N) and let d=q_ampl_mltplr.
    %                    Compute D such that D/N=(1-F(d))/2, where F() is Gaussian
    %                    distribution with zero mean and variance of one.
    %                    Then compute the estimate of the Gaussian variance
    %                    from x(1+D),...,x(N-D) by:
    %                      variance = var(x(1+D:N-D))/C((F(A)-F(-A))-2*A*f(A))
    %                    where
    %                      C = 2*(F(d) - 0.5 - d*f(d)) =
    %                      erf(A)-2*A*exp(-A^2)/sqrt(pi)
    %                    where f() is the PDF corresponding to F and
    %                    A=d/sqrt(2)
    %                    
    %                    Note: If mean is not specified or [], the mean is
    %                    computed as the mean of x(1+D:N-D)
    %                
    %
    %   q_cls     - (optional) class name of output.  Must be a subclass of
    %               QuantMeasurements 
    % Output
    %   qmsr - an object of type QuantMeasurements
    function  qmsr= ...
            quantize(obj, msrmnts, params, q_cls)
        if nargin < 4
            q_cls = 'QuantMeasurements';
        end
        qmsr = eval(q_cls);
        
        n_msr = length(msrmnts);
        if n_msr == 0;
            qmsr.bin_numbers = [];
            qmsr.mean_msr = 0;
            qmsr.stdv_msr = 0;
            qmsr.saved = [];
            qmsr.n_no_clip = 0;
            qmsr.n_bins = 0;
            return
        end
        
        n_bins = [];
        intvl = [];

        % Process params
        if isfield(params,'n_no_clip')
            n_no_clip = params.n_no_clip;
            if n_no_clip > n_msr;
                n_no_clip = n_msr;
            end
        else
            n_no_clip = 0;
        end
        qmsr.n_no_clip = n_no_clip;
        n_clippable = n_msr - n_no_clip;
        
        if isfield(params,'q_unit') && ~isempty(params.q_unit )
            q_unit = params.q_unit;
        else
            q_unit = obj.q_wdth_unit;
        end
        
        if isfield(params,'mean') && ~isempty(params.mean )
            qmsr.mean_msr = params.mean;
        elseif ~isfield(params,'stdv') || isempty(params.stdv) || params.stdv ~= -1
            if n_clippable > 0
                qmsr.mean_msr = mean(msrmnts(n_no_clip+1:end));
            else
                qmsr.mean_msr = 0;
            end
        end
        
        if isfield(params,'stdv') && ~isempty(params.stdv)
            if params.stdv ~= -1
                qmsr.stdv_msr = params.stdv;
            else
              % Special case of params.stdv == -1
              A=obj.q_ampl_mltplr/sqrt(2);
              E=erf(A);
              D=round(n_clippable*0.5*(1-E));
              msrs = sort(msrmnts(n_no_clip+1:end));
              var_msrs = var(msrs(1+D:end-D))/(E-(2/sqrt(pi))*A*exp(-A*A));
              qmsr.stdv_msr = sqrt(var_msrs);
              
              if ~isfield(params,'mean') || isempty(params.mean )
                qmsr.mean_msr = mean(msrs(1+D:end-D));
              end
              
              intvl = obj.calcIntvl(q_unit);
              n_bins = ceil((msrs(end-D)-msrs(1+D))/(2*intvl))*2+1;
            end
        elseif n_clippable > 1
            qmsr.stdv_msr = sqrt(var(msrmnts(n_no_clip+1:end)));
        else
            qmsr.stdv_msr = 0;
        end
        
        % Compute interval and amplitude
        [intvl, n_bins, ampl] = ...
            obj.calcParams(q_unit, qmsr.stdv_msr, n_bins, intvl);
        qmsr.n_bins = n_bins;

        % Do quantization - compute bin numbers
        offset = ampl - qmsr.mean_msr;
        qmsr.bin_numbers = ceil((offset + msrmnts )/intvl);

        % Find values to clip
        bin_clippable = qmsr.bin_numbers(n_no_clip+1:end);
        too_high = n_no_clip + find(bin_clippable > n_bins);
        too_low = n_no_clip + find(bin_clippable < 1);
        if iscolumn(msrmnts)
            out_bins = [too_high; too_low];
        else
            out_bins = [too_high too_low];
        end
        
        % Save values if necessary
        if obj.save_clipped
            qmsr.saved = qmsr.bin_numbers(sort(out_bins));
            hi_out = find(qmsr.saved > n_bins);
            qmsr.saved(hi_out) = qmsr.saved(hi_out) - n_bins;
        else
            qmsr.saved = [];
        end
        
        % clip
        qmsr.bin_numbers(out_bins) = n_bins+1;
    end
    
    % unquantize - reconstruct values from quantization bin numbers
    % Input:
    %   obj - the quantizer object
    %   qmsr - an object of type QuantMeasurements
    %   q_unit - (optional) overrides obj.q_wdth_unit
    % output
    %   msrmnts - Reconstructed measurements
    %   clipped_indices - Indices of measurement which could not be reconstructed
    %                  because they were clipped.
    %   
    function [msrmnts, clipped_indices] = unquantize(obj, qmsr, q_unit)
        n_msr = length(qmsr.bin_numbers);
        
        if nargin < 3
            q_unit = obj.q_wdth_unit;
        end
        
        % Compute interval and amplitude
        [intvl, n_bins, ampl] = obj.calcParams(q_unit, qmsr.stdv_msr, qmsr.n_bins);
        n_no_clip = double(qmsr.n_no_clip);
        if n_no_clip > n_msr
            n_no_clip = n_msr;
        end
       
        clipped_indices = n_no_clip + ...
            find(qmsr.bin_numbers(n_no_clip+1:end) > n_bins);

        % Do unquantization
        offset = ampl + 0.5 *intvl - qmsr.mean_msr;
        msrmnts = (double(qmsr.bin_numbers) * intvl) - offset;
        
        if UniformQuantizer.correct_gaus_unquant && qmsr.stdv_msr > 0
            msrmnts = obj.correctGausUnquant(msrmnts, qmsr, intvl, n_bins);
        end
        
        % Fill clipped measurements with saved measurements, if available
        n_saved = length(qmsr.saved);
        if n_saved > 0
            if length(clipped_indices) < n_saved
                exception = MException('UniformQuantizer:unquantize:saved',...
                    'saved array too long');
                throw(exception);
            end
            hi_ind = find(qmsr.saved > 0);
            qmsr.saved(hi_ind) = qmsr.saved(hi_ind) + n_bins;
            msrmnts(clipped_indices(1:n_saved)) = (qmsr.saved * intvl) - offset;
            clipped_indices = clipped_indices(n_saved+1:end);
        end
        
        msrmnts(clipped_indices) = 0;
    end
    
    % binGaussProb computes bins probabilities for a Gaussian distribution
    % Input 
    %   obj - this object
    %   stdv_msr - Standard deviation of measurements
    %   q_unit - (optional) overrides obj.q_wdth_unit
    % Outupt
    %   probs - an array of bins probabilities
    %   out_prob - probability of outlier.
    % Explanation
    % The number of bins, n_bins is odd: n_bins=2*n1+1.  Each bin corresponds
    % to an interval of width intvl.  Let the bins be numbered
    % -n1,...,0,...,n1, then the middle of bin 0 corresponds to the mean. The
    % upper boundary of the k-th bin is s(k)=intvl*(k+0.5).  Let F(x) be the
    % Gaussian probability distribution function with mean 0 and standard
    % deviation stdv.  Then the probability of bin k is F(s(k))-F(s(k-1)).
    % Because of symmetricity we need to compute F(s(k)) only for k=0,...,n1.
    % The computation is done using the erf function (
    % erf(y)=(2/sqrt(pi))*integral(0 to y of exp(-t*t) dt) ).  It is easy to
    % verify that erf(s/(stdv*sqrt(2)))=1+2F(s).  Thus if we compute erf at
    % s(k) we can obtain the bin probabilities by convolving with [-0.5,0.5].
    % special care has to be given to the first bin, where the low boundary is
    % actually at the center of the bin (hence F(s(0))=0.5) and the last bin
    % representing the high outliers where the high boundary is infinitiy,
    % F(inf)=1.
    function [prob, out_prob] = binGaussProb(obj, stdv_msr, n_bins, q_unit)
        if nargin < 4
            q_unit = obj.q_wdth_unit;
        end
        
        % Insure positive standard deviation
        [intvl, n_bins, ~] = obj.calcParams(q_unit, stdv_msr, n_bins);
        n1 = (n_bins-1)/2;
        
        % pts is the array at which erf is computed.
        pts = ((0:n1)+0.5)*(intvl/(stdv_msr*sqrt(2)));
        
        % cdf is the array of 1+2F(s(k)).  The first point (implicit 0) and 
        % the last point correspond to 0 and infinity
        cdf = [erf(pts), 1];
        prb = conv(cdf, [0.5,-0.5]);
        out_prob = 2*prb(end-1);
        prb0 = 2*prb(1);
        prb = prb(2:end-2);
        prob = [fliplr(prb), prb0, prb];
    end
    
    % encode - implementation of abstract function from CodeElement
    function len = encode(obj, code_dst, ~)
        len = code_dst.writeUInt([obj.save_clipped]);
        if ischar(len)
            return;
        end
        n = code_dst.writeNumber(...
            [obj.q_wdth_mltplr, obj.q_wdth_unit, obj.q_ampl_mltplr]);
        if ischar(n)
            len = n;
            return;
        end
        len = len + n;
    end
    
    % decode - implementation of abstract function from CodeElement
    function len = decode(obj, code_src, ~, cnt)
        if nargin < 4
            cnt = inf;
        end
        [vals, n] = code_src.readUInt(cnt, [1,1]);
        if ischar(vals) || (isscalar(vals) && (vals==-1));
            len = vals;
            return;
        end
        len = n;
        sv_clp = vals(1);
        
        [vals, n] = code_src.readNumber(cnt-len, [1,3]);
        if ischar(vals)
            len = vals;
            return;
        elseif length(vals)==1 && (isscalar(vals) && (vals==-1))
            len = 'Unexpected end of data';
            return
        end
        len = len + n;
        obj.save_clipped = sv_clp;
        obj.q_wdth_mltplr = vals(1);
        q_unit = vals(2);
        obj.q_ampl_mltplr = vals(3);
        obj.q_wdth = obj.calcIntvl(q_unit);
        obj.q_wdth_unit = q_unit;
    end
    
end

methods (Access=protected)
    
    % calcParams - calculate quantizer parameters
    function [intvl, n_bins, ampl] = ...
            calcParams(obj, q_unit, stdv_msr, n_bins, intvl)
        if nargin < 5 || isempty(intvl)
            intvl = obj.calcIntvl(q_unit);
        end
        if nargin < 4 || isempty(n_bins)
            max_bin = round((stdv_msr * obj.q_ampl_mltplr)/intvl);
            n_bins = 2*max_bin + 1;
        else
            max_bin = floor(n_bins-1)/2;
        end
        ampl = (max_bin + 0.5) * intvl;
    end
    
    % calcIntvl - compute quantization interval
    function wdth = calcIntvl(obj, q_unit)
        if obj.q_wdth_mltplr == 0 || nargin <2 || q_unit == 0
            wdth = 1;
        elseif q_unit == obj.q_wdth_unit
            wdth = obj.q_wdth;
        else
            wdth = obj.q_wdth_mltplr * q_unit;
        end
    end    
end

methods (Static)
    % Compute the source entropy, assuming that it is Guassian with sigma=1
    % (otherwise step has to be divided by sigma).
    %   Input
    %     max_bin - maximal bin number. Bin numbers range from -max_bin+1
    %         to max_bin, where max_bin-1 is the highest unsaturated bin
    %         and max_bin indicates saturation.  max_bin can be a row
    %         vector, in which case results are calculate for each value in
    %         the array.
    %     step - quantization step size. can be a column vector, in which
    %     case results are calculated for each column in the vector
    %     stddev - Standard deviation of measurements (optional, default=1)
    %   Output.  All outuputs are arrays of the same size as max_bin, with
    %   corresponding values.
    %
    %     entrp - The entropy of the quantization labels.
    %     use_prob - probability that the label is not saturated
    %     use_entrp - entropy per used measurment:  entrp/use_prob.
    function [entrp, use_prob, use_entrp] = compGausEntropy(max_bin,step, stddev)
        if nargin > 2
            step = step /stddev;
        end
        n_step = length(step);
        n_max = max(max_bin);
        n_vals = length(max_bin);
        
        % Allocate arrays;
        entrp = zeros(n_step,n_vals);
        if nargout > 1
            use_prob = zeros(n_step,n_vals);
            if nargout > 2
                use_entrp = zeros(n_step,n_vals);
            end
        end
        
        % Compute values
        for istp = 1:n_step
            pts = (step(istp)/sqrt(2))*(0.5:1:(n_max-0.5));
            vals = erf(pts);
            dff = diff([0 vals]);
            dff2 =  dff;
            dff2(2:end) = dff2(2:end)*0.5;
            ents = -dff .* log(dff2);
            ents(isnan(ents))=0;
            
            for k=1:n_vals
                k_max = max_bin(k);
                used_p = vals(k_max);
                unused_p = 1- used_p;
                ent_sat = - unused_p * log(unused_p);
                if isnan(ent_sat); ent_sat = 0; end
                entrp(istp,k) = sum(ents(1:k_max)) + ent_sat;
                if nargout > 1
                    use_prob(istp,k) = used_p;
                    if nargout > 2
                        use_entrp(istp,k) = entrp(istp,k)/used_p;
                    end
                end
            end
        end
    end
    
    % Compute the source entropy, assuming that it is Guassian with sigma=1
    % (otherwise step has to be divided by sigma).
    %   Input
    %     max_bin - maximal bin number. Bin numbers range from -max_bin+1
    %         to max_bin, where max_bin-1 is the highest unsaturated bin
    %         and max_bin indicates saturation.  max_bin can be a row
    %         vector, in which case results are calculate for each value in
    %         the vector.
    %     ampl - quantization amplitude. It can be a column vector, in which
    %         case results are calculate for each value in
    %         the vector. 
    %   Output.  All outuputs are arrays of the same size as max_bin, with
    %   corresponding values.
    %
    %     entrp - The entropy of the quantization labels.
    %     use_prob - probability that the label is not saturated
    %     use_entrp - entropy per used measurment:  entrp/use_prob.
    function [entrp, use_prob, use_entrp] = compGausEntropyAmpl(max_bin,ampl)
        entrp = zeros(length(ampl), length(max_bin));
        use_prob = entrp;
        use_entrp = entrp;
        for j=1:length(ampl)
            for k=1:length(max_bin)
                [e,u,ue] =...
                    UniformQuantizer.compGausEntropy(max_bin(k), ...
                    ampl(j)/(0.5+max_bin(k)));
                entrp(j,k) = e;
                use_prob(j,k) = u;
                use_entrp(j,k) = ue;
            end
        end
    end
    
    
    % Compute the entropy, per used bin, assuming that the source is 
    % Guassian with sigma=1(otherwise step has to be divided by sigma).
    %   Input
    %     max_bin - maximal bin number. Bin numbers range from -max_bin+1
    %         to max_bin, where max_bin-1 is the highest unsaturated bin
    %         and max_bin indicates saturation.  max_bin can be a row
    %         array, in which case results are calculate for each value in
    %         the array.
    %     step - quantization step size.
    %   Output.  All outuputs are arrays of the same size as max_bin, with
    %   corresponding values.
    %     use_entrp - entropy per used measurment/ array of the same size 
    %                 as max_bin, with corresponding values.
    function use_entrp = compGausEntropyPerUsed(max_bin,step)
        [~,~,use_entrp] = UniformQuantizer.compGausEntropy(max_bin,step);
    end
    
    % Correct the unquantized measurements by taking into account that the
    % distribution in each bin is Gaussian, not uniform
    % Input:
    %   msrments - measurements computed based on Gaussian assumption
    %   qmsr - The unquantized measurements, assuming uniform behavior
    %   intvl - step size
    %   n_bins - number of unsaturated bins
    % Output
    %   msrmnts - corrected measurements
    function msrmnts = correctGausUnquant(msrmnts, qmsr, intvl, n_bins)
        max_bin = (n_bins-1)/2;
        step = intvl/qmsr.stdv_msr;
        step2 = 0.5*step*step;
        denom = 0.5 * diff(erf((step/sqrt(2))*(0.5:max_bin+0.5)));
        inds = 1:max_bin;
        nom = (sqrt(2/pi))*exp(-step2*(inds.*inds+0.25)).*sinh(step2*inds);
        crct = ((nom./denom) - inds*step)*qmsr.stdv_msr;
        crct = [-crct(max_bin:-1:1), 0, crct];
        
        n_no_clip = double(qmsr.n_no_clip);
        bin_inds = n_no_clip + find(qmsr.bin_numbers(n_no_clip+1:end) <= n_bins);
        msrmnts(bin_inds) = msrmnts(bin_inds) + crct(qmsr.bin_numbers(bin_inds))';
    end

end
end

