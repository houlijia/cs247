classdef FileReportInfo <FrmReportInfo
  % Used by CSVicCodec to report file performance
  properties
  end
  
  methods
    function report(obj, prefix)
      if ~obj.n_frms
        fprintf('%s   no frames in %.f sec\n', prefix, sts_dur);
        return
      end
      sts = obj.getStatus();
      
      bpxl_str = obj.getBpxlStr(sts);
      psnr_str = obj.getPsnrStr(sts);
      qmsr_str = obj.getQMsrsStr(sts);
      
      fprintf('%s    %d frames, in %.1f sec, %.3f sec/fr%s%s%s.\n', prefix,...
        sts.n_frms, sts.dur, sts.dur/sts.n_frms, qmsr_str, bpxl_str,psnr_str);
    end
    
  end
  
end

