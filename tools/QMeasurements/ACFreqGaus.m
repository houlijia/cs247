classdef ACFreqGaus < ACFreq
  % ACFreqGaus ACFreq using Gaussian approximation
  
  methods
    function obj = ACFreqGaus(stdv_msr, n_bins, intvl, nsq)
      % Constructor. Input arguments:
      %   stdv_msr - Standard deviation of measurements
      %   intvl - the quantization interval
      %   n_bins - number of bins in range (not including outliers)
      %   nsq - length of the sequence

      frq = ACFreqGaus.calcFreqGaus(stdv_msr, n_bins, intvl);
      obj = obj@ACFreq(frq, nsq);
    end
    
    function len = encode(~, ~)
      % Nothing to encode
      len = 0;
    end
    

    function md = getMode(~)
      md = ACFreq.MODE_GAUS;
    end
      
    function len = histLength(~)
      len = 0;
    end
  end
  
  properties(Constant, Access=private)
    PROB_MLTPLR = 1e6;
    min_stdv = 1e-10;   % Minimum standard deviation
  end
  

  methods(Access=private, Static)
    function frq = calcFreqGaus(stdv, nbn, intvl)
      % Compute the frequency of labels assuming Gaussian distribution
      % Input:
      %   stdv - standard deviation of the Gaussian distribution
      %   nbn - number of unsaturated bins
      %   intvl - length of the bin interval
      % Output:
      %   frq - array of nbn+1 positive integers, containing estimated
      %     frequencies.
      stdv = max(ACFreqGaus.min_stdv, stdv);
      [prob, prob_out] = ACFreqGaus.binGaussProb(stdv, intvl, nbn);
      frq = round(([prob, prob_out]' * ACFreqGaus.PROB_MLTPLR) + 1);
    end
    
    function [prob, out_prob] = binGaussProb(stdv_msr, intvl, n_bins)
      % binGaussProb computes bins probabilities for a Gaussian distribution
      % Input
      %   stdv_msr - Standard deviation of measurements
      %   intvl - the quantization interval
      %   n_bins - number of bins in range (not including outliers)
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
      
      % Insure positive standard deviation
      n1 = (double(n_bins)-1)/2;
      
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
    
  end
end

