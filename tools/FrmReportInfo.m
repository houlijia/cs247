classdef FrmReportInfo < handle
  % Used by CSVidCodec to report frames
  
  properties
    n_frms;
    f_bgn;
    f_end;
    n_bytes;
    n_pxls;
    n_msrs;
    sum_exp_psnr;
    psnr_cnt = 0;
    sum_msrs_sqr;
    sum_msrs_err_sqr;
    start_time;
  end
  
  methods
    function obj = FrmReportInfo()
      obj.reset();
    end
    
    function status = reset(obj)
      if nargout > 0
        status = obj.getStatus();
      end
      obj.start_time = tic();
      obj.n_frms = 0;
      obj.n_bytes = 0;
      obj.n_pxls = 0;
      obj.n_msrs = 0;
      obj.f_bgn = [];
      obj.psnr_cnt = 0;
      obj.sum_exp_psnr = 0;
      obj.sum_msrs_sqr = 0;
      obj.sum_msrs_err_sqr = 0;
    end
    
    function status = getStatus(obj)
      status = struct(...
        'n_frms', obj.n_frms,...
        'f_bgn', obj.f_bgn,...
        'f_end', obj.f_end,...
        'n_bytes', obj.n_bytes,...
        'n_pxls', obj.n_pxls,...
        'n_msrs', obj.n_msrs,...
        'sum_exp_psnr', obj.sum_exp_psnr,...
        'psnr_cnt', obj.psnr_cnt,...
        'sum_msrs_sqr', obj.sum_msrs_sqr,...
        'sum_msrs_err_sqr', obj.sum_msrs_err_sqr,...
        'dur', toc(obj.start_time)...
        );
    end
    function update(obj, n_pxls, blks_data)
      obj.n_pxls = obj.n_pxls + n_pxls;
      if isempty(obj.f_bgn)
          [b_bgn,~,~,~,~,~] = blks_data(1).enc_info.vid_blocker.blkPosition(...
            blks_data(1).enc_info.blk_indx);
          obj.f_bgn = b_bgn(1,3);
      end
      if blks_data(end).last_in_frm
          [~,b_end,~,~,~,~] = blks_data(end).enc_info.vid_blocker.blkPosition(...
            blks_data(end).enc_info.blk_indx);
          obj.f_end = b_end(1,3);
          obj.n_frms = obj.f_end - obj.f_bgn + 1;
      end
    end
    
    function updateBytes(obj, n_b, n_m)
      obj.n_bytes = obj.n_bytes + n_b;
      obj.n_msrs = obj.n_msrs + n_m;
    end
    
    function updatePsnr(obj, rdata)
      for idx=1:numel(rdata)
        if isfield(rdata{idx},'frms_psnr')
          obj.psnr_cnt = obj.psnr_cnt + 1;
          obj.sum_exp_psnr = obj.sum_exp_psnr + exp(rdata{idx}.frms_psnr);
        end
      end
    end
    
    function updateMsrsErr(obj, orig_msrs, q_msrs)
      obj.sum_msrs_sqr = obj.sum_msrs_sqr + ...
        sum(double(orig_msrs) .^ 2);
      obj.sum_msrs_err_sqr = obj.sum_msrs_err_sqr + ...
        sum(double(q_msrs - orig_msrs) .^ 2);
    end
    
    function report(obj, prefix)
      if ~obj.n_frms
        return
      end
      sts = obj.reset();
      
      bpxl_str = obj.getBpxlStr(sts);
      psnr_str = obj.getPsnrStr(sts);
      qmsr_str = obj.getQMsrsStr(sts);
      
      fprintf('%s frames %d:%d: %.3f sec/fr%s%s%s.\n', prefix,...
        sts.f_bgn, sts.f_end, sts.dur/sts.n_frms, qmsr_str, bpxl_str, psnr_str);
    end
    
  end
  
  methods (Access=protected, Static)
    function str = getBpxlStr(sts)
      if sts.n_bytes > 0
        bpxl = (double(sts.n_bytes) * 8)/double(sts.n_pxls);
        str = sprintf(', %.3f b/pxl', bpxl);
      else
        str = '';
      end
      if sts.n_msrs > 0
        b_msr = (double(sts.n_bytes) * 8)/double(sts.n_msrs);
        str = [str sprintf(', %.3f b/msr', b_msr)];
      end
    end
    
    function str = getPsnrStr(sts)
      if sts.psnr_cnt > 0
        str = sprintf(', PSNR =%.1f dB',...
          log(sts.sum_exp_psnr/sts.psnr_cnt + 1E-20));
      else
        str = '';
      end
    end
    
    function str = getQMsrsStr(sts)
      if sts.sum_msrs_sqr > 0
        msrs_snr = ...
          10*log10((1e-3 + sts.sum_msrs_sqr/1e-3 + sts.sum_msrs_err_sqr));
        msrs_snr = min(99, max(-9, msrs_snr));
        str = sprintf(', Q.Msrs SNR=%4.1f dB', msrs_snr);
      else
        str = '';
      end
    end
  end
  
end

