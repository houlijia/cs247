function files_def = brief_tcp_files_def( addr, opts, names )
  % brief_tcp_files_def Sgenerates a struct for defining I/O for brief_tcp_enc
  % and brief_tcp_dec
  %
  % INPUT Arguments (all optional)
  %   addr - addresses and ports to use, in the format
  %          <host>:<port>[::<lcl_host>>[:<lcl_port>]] where the default is
  %          localhost:9000
  %   opts - a struct with the following optional fields
  %          sndr_is_srvr (logical):, Default: false
  %          cnct_timeout: Time (sec) server waits for connection, 0= indefinite.
  %                        Default: 60
  %          recv_timeout: Timeout (sec) for receiver to wait for data. Default:
  %                        30
  %          linger_timeout: Time (seconds) for sender to linger after
  %                          finishing, while closing TCP connection, to make
  %                          sure that the receiver received all data that is
  %                          in the pipeline. -1 means use system default. Any
  %                          non-negative value is rounded up to an integer
  %                          value, and the sender wait up to that time and then
  %                          performs a hard reset. Default: 300
  %    
  %   names - base names of files to be used by sender. Default: \
  %           {'foreman_cif_300'', 'news_cif_300'}
  
  ref_opts = struct('sndr_is_srvr', false,...
    'cnct_timeout', 60,...
    'recv_timeout', 30,...
    'linger_timeout', 300);
    
  if nargin < 3
    names = {'foreman_cif_300', 'news_cif_300'};
  end
  
  if nargin < 2
    opts  = ref_opts;
    if nargin < 1
      addr = 'localhost:9000';
    end
  else
    flds = fieldnames(ref_opts);
    for k =  1:length(flds);
      fld = flds{k};
      if ~isfield(opts, fld)
        opts.(fld) = ref_opts.(fld);
      end
    end
  end
  
  opts.addr = addr;
  
  files_def = struct(...
    'names', {names},...
    'types', struct(...
       'txt','',...
       'input','.json',...
       'mat','.mat',...
       'output','.o.yuv',...
       'inp_anls','i.anls',...
       'enc_vid', {{'.csvid', opts}},... 
       'cnv_enc', 'cnv_svid',...
       'cnv_mat', '.cnv_mat',...
       'enc_pre_diff', '.edff.yuv',...
       'dec_pre_diff', '.ddff.yuv',...
       'tst_pre_diff', '.ddfo.yuv',...
       'err_pre_diff', '.ddfe.yuv'),...
     'input_dir', pwd(),...
     'output_dir', [pwd() filesep '..' filesep 'output' filesep 'tcp_*'],...
     'case_dir', '*.case',...
     'dec_dir', 'dec_*');
    

end

