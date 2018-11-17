function build_tcp_sock_mex(varargin)
  [mpath,~,~] = fileparts(mfilename('fullpath'));
  cd(mpath)
  
%   opts = {'-v'};
  opts = {};
  common_files={'tcp_sock_mex.c', 'tcp_sock_io.c', 'ip_util.c', 'ip_name.c', 'timeval.c'};
  
  if ispc()
    libs={'-lWs2_32', '-lwinmm'};
  else
    libs = {};
  end
  
  if nargin < 1
    targets = {'openTCPSocketServer', 'openTCPSocketClient', 'closeTCPSocket', ...
      'sendTCPSocket', 'recvTCPSocket', 'undefSocket'};
  else
    targets = varargin;
  end
  
  for k=1:length(targets)
    tgt = [targets{k} '_mex.c'];
    args = [opts {tgt} common_files libs];
    mex(args{:});
  end
end
