classdef SensingMatrixLclDFT < SensingMatrixSqrLclRnd & SensingMatrixDFT
  %SensingMatrixLclWH Same as SensingMatrixWH, except that the
  %randomization is local
  
  properties
  end
  
  methods
    function obj = SensingMatrixLclDFT(varargin)
      obj = obj@SensingMatrixSqrLclRnd();
      obj = obj@SensingMatrixDFT(varargin{:});
    end
    
    function set(obj, varargin)
      obj.set@SensingMatrixDFT(varargin{:});
    end
    
%     function y = multVec(obj,x)
%       y = multVec@SensingMatrix(obj,x);
%       
%       if ~obj.is_transposed
%         % testing
%         m = obj.n_rows;
%         n = size(x,1);
%         sm = sum(x);
%         mn = sm/n;
%         u = zeros(obj.sqr_order, size(x,2));
%         for k=1:size(u,2)
%           u(1:n,k) = obj.PR(1:n,1) .* (x(:,k)-mn(k));
%         end
%         v =obj.multSqr(u);
%         
%         vc = complex([v(1:2:end);v(2)],[0;v(4:2:end);0]);
%         [tlist, cmsrs, nrl] = obj.sortMsrsList(y,2:m);
%         if ~isequal(cmsrs, vc(tlist+1)) || nrl ~= 1
%           error('soreMsrsList wrong');
%         end
%         
%         u2 = u(:,1).*u(:,1);
%         U2 = obj.multSqr(u2);
%         U2C = complex([U2(1:2:end);U2(2)],[0;U2(4:2:end);0]);
%         
%         w = [vc ; conj(vc(end-1:-1:2))];
%         W = real(fft(w));
%         U2test=ifft(W .* W);
%         U2test = U2test(1:length(U2C))/obj.sqr_order;
%         
%         err = norm(U2test - U2C);
%         if err > 1e-8 * norm(U2C)
%           error('not matching');
%         end
% 
%       end
%     end
    
    function n_no_clip=nNoClip(obj)
      n_no_clip = obj.nNoClip@SensingMatrixSqrLclRnd();
    end
    
    function dc_val = getDC(obj,msrs)
      dc_val = obj.getDC@SensingMatrixSqrLclRnd(msrs);
    end
    
    % Get an array of pseudo-measurements which correspond to specific offsets.
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
    %           obj.IPL(inp_list(j))-obj.IPL(inp_list(i)) =
    %              ofsts(k) - ofst(n)  mod(obj.sqr_order)
    %
    %  Note: If params is present and has the fields ofsts_list
    %        and nghbr_list, then after computing ofst_msrs it is
    %        modified by a call
    %    ofst_msrs = obj.getEdgeMsrmnts(ofst_msrs,
    %                                   arams.ofsts_list,
    %                                    params.nghbr_list);
    %        or something functionally equivalent.
    function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts, msrs,...
        inp_list, params)
      if nargin < 5
        params = struct();
      end
      if ~isfield(params, 'nrm_exp')
        params = struct('nrm_exp',1);
      end
      if ~isnumeric(inp_list)
        inp_list = 1:obj.nRows();
      end
      
      if ~isempty(obj.zeroed_rows)
        [~,zr,~] = intersect(inp_list, obj.zeroed_rows);
        inp_list(zr) = [];
      end
      
      % Get all measurements except for the DC one, which has a sepcial
      % meaning.
      [tlist, cmsrs] = obj.sortMsrsList(msrs, inp_list);
      
      % Remove DC, since the DC measurement was
      % overloaded with a special meaning.
      i0 = find(tlist==0);
      tlist(i0) = [];
      cmsrs(i0) = [];

      [vmsrs, v_list] = auto_conv_sym(cmsrs, tlist);
      [cnt_msrs, cnt_list] = auto_conv_sym(true(length(tlist),1),tlist);
      
      % Eliminate small counts. In the calculation ignore the 0 entry which
      % is very large
      if length(cnt_msrs) > 1
        thresh = 1.0;
        c_ind = find(cnt_msrs >= thresh*mean(cnt_msrs(2:end)));
        cnt_msrs = cnt_msrs(c_ind);
        cnt_list = cnt_list(c_ind);
      end
      
      [c_indcs, v_ind, c_ind] = intersect(v_list, cnt_list);
      vmsrs = vmsrs(v_ind);
      cnt_msrs = cnt_msrs(c_ind);
      % Normalize by number of elements 
      vmsrs = vmsrs ./cnt_msrs;
      
      % put self symmetric values first
      if c_indcs(end) == obj.sqr_order/2
        if c_indcs(1) == 0
          c_indcs = [c_indcs(1); c_indcs(end); c_indcs(2:end-1)];
          vmsrs = [vmsrs(1); vmsrs(end); vmsrs(2:end-1)];
          nrl = 2;
        else
          c_indcs = [c_indcs(end); c_indcs(1:end-1)];
          vmsrs = [vmsrs(end); vmsrs(1:end-1)];
          nrl = 1;
        end
      elseif c_indcs(1) == 0
          nrl = 1;
      else
        nrl = 0;
      end
      
%       % Discard very small values
%       abs_u = abs(vmsrs);
%       avg = (norm(abs_u(1:nrl),1) + 2*norm(abs_u(nrl+1:end),1))/...
%         (2*length(abs_u)-nrl);
%       eps = 0.1;
%       h_indcs = find(abs_u > eps * avg);
%       c_indcs = c_indcs(h_indcs);
%       vmsrs = vmsrs(h_indcs);      
      
%       % Compute scalers for minimum square error
%       n_indcs = length(c_indcs);
%       v2 = vmsrs .* conj(vmsrs);
%       v2av = (sum(v2(1:nrl)) + 2*sum(v2(nrl+1:end)))/obj.sqr_order;
%       v2c = 2*(v2 - v2av)/((2*n_indcs-nrl)^2);
%       v2c(nrl+1:end) = 2*v2c(nrl+1:end);
%       scaler = zeros(n_indcs,1);
%       for k=1:n_indcs
%         k_shft = mod([k-c_indcs;k+c_indcs],obj.sqr_order);
%         [k_indcs, ~,~] = intersect(c_indcs, k_shft);
%         [k_cnts, k_indcs] = auto_conv_sym(true(length(k_indcs),1),k_indcs);
%         [~, k_ptr, c_ptr] = intersect(k_indcs, c_indcs);
%         k_cnts = k_cnts(k_ptr);
%         k_msrs = v2c(c_ptr);
%         scaler(k) = v2(k)/(v2(k) + dot(k_msrs, k_cnts));
%       end
%       vmsrs = vmsrs .* scaler;
      
      % correct for elements where conjugate is not self
      vmsrs(nrl+1:end) = vmsrs(nrl+1:end)  * (2 ^ (1/params.nrm_exp));
       
      wgts = obj.getShiftWgts(obj.log2order);
      
      if isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
        rsize = obj.convertToRealSize([length(c_indcs),...
          length(params.ofsts_list)], nrl);
      else
        rsize = obj.convertToRealSize([length(c_indcs),length(ofsts)],nrl);
      end
      ofst_msrs = zeros(rsize,'single');
      
      m_step = 1024;
      r_bgn = 1;      
      for m_bgn=1:m_step:length(c_indcs)
        m_end = min(m_bgn+m_step-1,length(c_indcs));
        
        wgt_ind = reshape((mod(c_indcs(m_bgn:m_end)*ofsts(:)', obj.sqr_order)+1),...
          [m_end-m_bgn+1, size(ofsts,1)]);
        ofst_msrs_c = (vmsrs(m_bgn:m_end)*ones(1,length(ofsts)))...
          .* reshape(wgts(wgt_ind),size(wgt_ind));
        
        if isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
          ofst_msrs_c  = obj.getEdgeMsrmnts(ofst_msrs_c, ...
            params.ofsts_list, params.nghbr_list);
        end
        
        r_sz = obj.convertToRealSize(size(ofst_msrs_c),nrl);
        r_end = r_bgn + r_sz(1) - 1;
        ofst_msrs(r_bgn:r_end, :) = obj.convertToReal(ofst_msrs_c,nrl);
        nrl = max(nrl-m_step,0);
        r_bgn = r_end + 1;
      end
      
      % Compute circular convolution (modulo obj.sqr_order) of a conjugate 
      % symmetric sequence with itself. Since both input and output are
      % specified only for indices 0,...,N/2. FFT is used to compute the 
      % convolution faster.
      %   Input:
      %     v - non zero values in input array. Can be logical, real or
      %         complex.
      %     v_ind - indices to which v_ind corresponds (starting from 0).
      %             It is necessary to supply only the indices between 0 to
      %             N/2.
      %   Output:
      %     u - non-zero values of output (real, since input is conjugate
      %         symmetric). If input is logical the output is rounded to
      %         the nearest integer.
      %     u_ind - indices corresponding to values of u, in the range of 0
      %             to n/2. The indices corresponding to 0 and N/2 (i.e.
      %             the self symmetic ones), if present are first
      %   
      function [u, u_ind]=auto_conv_sym(v, v_ind)
        if obj.sqr_order == 1
          u = v * v;
          u_ind = v_ind;
          return;
        end
        
        mid = obj.sqr_order/2;
        w = zeros(mid+1,1);
        w(v_ind+1) = v;
        if islogical(v) || isreal(v)
          w = [w ; w(end-1:-1:2)];
        else
          w = [w ; conj(w(end-1:-1:2))];
        end
        
        % Since in our case the input is conjugate symmetric, the
        % output is real
        W = real(fft(w));
        
        u=ifft(W .* W);

        u = u(1:mid+1);
        if islogical(v)
          u_ind = find(u >0.5);
          u = round(u(u_ind));
        else
          if isreal(v)
            u = real(u);
          end
          u_ind = find(u);
          u = u(u_ind);
        end
        u_ind = u_ind-1;
        if u_ind(end) == mid
          if u_ind(1) == 0
            u_ind = [0; u_ind(end); u_ind(2:end-1)];
            u = [u(1); u(end); u(2:end-1)];
          else
            u_ind = [u_ind(end); u_ind(1:end-1)];
            u = [u(end); u(1:end-1)];
          end
        end
        
      end
    end
  end
  
  methods(Access=protected)
    function [PL, PR] = makePermutations(obj, order, ~)
      PL = obj.makeRowSlctPermutation(order, obj.n_rows);
      PR = obj.makeLclPR(order);
    end
  end
  
end


