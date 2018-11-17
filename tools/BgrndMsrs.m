classdef BgrndMsrs < handle
  %BgrndMsrs estimates the background in a sequence of temporal blocks
  
  properties (SetAccess=immutable)
    n_blk;     % n_blk - number of consecutive temporal blocks.
    max_avg;   % maximumm number o elements to use in averagig. If cnt >
               % max_avg we use a forgetting factor of 1/(max_avg+1).
    min_dcd;   % Minimum number of blocks needed to make a decision.
    thrsh;     % Threshold for determining background
  end

  properties
    bg_age;  % For how many blocks has the current background been valid
    mtrx;    % Sensing matrix used in each block (cell arrayof dimension [n_blk,1])
    
    % The following properties are of dimension [n_blk,2]. The first column
    % is the current for the current background. It is updated only if the
    % input vector has been declared as background. The second column is
    % the alternative background - it is updated by all blocks.
    cnt;         % number of measurement vectors seen per block.
    sum_sqr_wgts % sum square of weights (the weigts always add to 1)
    msr_avg;     % Average of normalized measurements (cell array)
    msr_msqr     % Mean sum square of normalized meaurements (cell array)
    is_bgrnd;  % decisions for each block
  end
  
  methods
    function obj = BgrndMsrs(n_b, mx_avg, mn_dcd, thr)
      obj.n_blk = n_b;
      if n_b == 0
        obj.n_blk = 0;
        return;
      end
      obj.max_avg = mx_avg;
      obj.min_dcd = mn_dcd;
      obj.thrsh = thr;
      obj.bg_age = 0;
      obj.mtrx = cell(n_b,1);
      obj.cnt = zeros(n_b, 2);
      obj.sum_sqr_wgts = zeros(n_b,2);
      obj.msr_avg = cell(n_b, 2);
      obj.msr_msqr = cell(n_b, 2);
      obj.is_bgrnd = false(n_b, 2);
    end
    
    % Process block.
    %   Input:
    %     b_indx - index of the block;
    %     mtrx - Sensing matrix used for the block'
    %     msrs - The measurments vector.
    %  Output:
    %    rslt: 1=background, 0=not background, -1=background changed
    %    age - for how many blocks has the current background been valid
    function [rslt, age] = checkBlk(obj, b_indx, mtrx, msrs)
      if ~obj.n_blk
        rslt = 0;
        age = 0;
        return
      end
      
      idx = mod(b_indx-1,obj.n_blk)+1;
      msrs = mtrx.normalizeMsrs(msrs);

      % Initialize if necessarty
      if isempty(obj.mtrx{idx})
        obj.mtrx{idx} = mtrx;
        obj.msr_avg(idx,:) = {zeros(length(msrs),1)};
        obj.msr_msqr(idx,:) = {zeros(length(msrs),1)};
      end
      
      for k=1:2
        % The statistic is computed as follows:
        % *  subtract the mean background from each measurement and square
        % *  Compute and divide by the variance for each measurements.
        %    Under the null hypothesis (it is background) we have squares
        %    of standard Gaussian variables, that is, random variables
        %    with mean 1 and variance 2
        % *  Average the results over all measurements. since we have a
        %    large number, the average is Gaussian with 1 mean and 2/nRows()
        %    variance.
        % *  Take the log. Since the variance is small, the log also
        %    behaves as a Gussian with the same variance but with zero
        %    mean.
        
        diff_msrs = msrs - obj.msr_avg{idx,k};
        var_msrs = (obj.msr_msqr{idx,k} - obj.msr_avg{idx,k})./...
          (1 - obj.sum_sqr_wgts(idx,k));
        ststc = log(mean(diff_msrs .^ 2 ./ var_msrs) + 1e-10);
        
        obj.is_bgrnd(idx,k) = (abs(ststc) <  obj.thrsh*2/length(msrs));
      end
      
      % perform update
      update(2);
      if obj.is_bgrnd(idx,1)
        update(1);
      end
      
      if all(obj.is_bgrnd(:,1)) && all(obj.cnt(:,1) >= obj.min_dcd)
        rslt = 1;
        age = obj.bg_age + 1;
      elseif all(obj.is_bgrnd(:,2)) && all(obj.cnt(:,2) >= obj.min_dcd)
        rslt = -1;
        age = 1;
        obj.cnt(idx,1) = obj.cnt(idx,2);
        obj.sum_sqr_wgts(idx,1) = obj.sum_sqr_wgts(idx,2);
        obj.msr_avg(idx,1) = obj.msr_avg(idx,2);
        obj.msr_msqr(idx,1) = obj.msr_msqr(idx,2);
        obj.is_bgrnd(idx,1) = obj.is_bgrnd(idx,2);
      else
        rslt = 0;
        age = obj.bg_age;
      end
      obj.bg_age = age;
      
      
      function update(k)
        if obj.cnt(idx,k) < max(obj.max_avg, obj.min_dcd)
          obj.cnt(idx,k) = obj.cnt(idx,k)+1;
        end
        if obj.cnt(idx,k) >= obj.max_avg
          wgt = 1/obj.max_avg;
        else
          wgt = 1/obj.cnt(idx,k);
        end
         
        obj.msr_avg{idx,k} = (1-wgt)*obj.msr_avg{idx,k} + wgt*msrs;
        obj.msr_msqr{idx,k} = (1-wgt)*obj.msr_avg{idx,k} + wgt*(msrs .^ 2);
        obj.sum_sqr_wgts(idx,k) = (1-wgt)^2 * obj.sum_sqr_wgts(idx,k) + wgt ^2;
        
        
      end
    end
  end
end

