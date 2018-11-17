classdef CodeSourceTCP < CodeSource
  % CodeSourceTCP is an implementation of CodeSource as a TCP Socket,
  % opened in server mode (listen).
  
  properties
    sock = [];   %socket
    lcl_addr;   % local address to bind to (default: '')
    lcl_port;   % Local port to lisen on
    dst_addr;   % IP address of the sender
    dst_port;   % port of sender
    cnct_timeout;
    recv_wait=0;  % Time to wait for data (0=wait indefinitely). read() 
                 % and readSeqs() time out after that time.
    buffer = [];   % A buffer of pre-read content for efficiency.
    buf_size = 1024;  % Size of minimal read attempt
    buf_done = 0;         % Number bytes already read from the buffer.
  end
  
  methods
    % Constructor.
    %   INPUT:
    %   timeout - (vector of length 1 or 2). timeout(1) specifies the
    %             listening time for connection. The values can be:
    %               -1: Open the TCP socket in client (connect) mode, so
    %                   timeout is not relevant.
    %               0: Open the TCP socket in server (listen) mode, and
    %                  wait indefinitely.
    %               >0: Open the TCP socket in server (listen) mode, and
    %                  wait timeout(1) seconds.
    %             If length(timeout)>1 then timeout(2) is the time to wait
    %             for the data during read. 0 indicates wait indefinitely.
    %             If length(timeout)==1, wait indefinitely.
    %   addr - a string specifying IP ddresses. It has the format
    %          <host>:<port>[::<l_host>[:<l_port>]]
    %          where <host> and <l_host> are IP addresses in dotted decimal
    %          notation or names, and <port>, <l_port> are port numbers in
    %          decimal notation (host order). If timeout(1)>=0, <l_host>
    %          and <l_port>, if present, are ignored, and <host>, <port>
    %          specify the address and port to listen on for connection
    %          (if <host> is empty INADDR_ANY is assumed). If timeout(1)<0,
    %          <host> and <port> specify the address to connect to and 
    %          <l_host>, <l_port>, if present, specify the local address to
    %          bind to. If <l_host> is empty or missing, INADDR_ANY is
    %          assumed; if <l_port> is empty or missing, 0 is assumed.
    function obj = CodeSourceTCP(timeout, addr)
      obj.cnct_timeout = timeout(1);
      is_srvr = (timeout(1) >= 0);
      if length(timeout) > 1
        obj.recv_wait = timeout(2);
      end
      
      tokens = regexp(addr,'^(.*)[:][:](.*)$', 'tokens');
      if ~isempty(tokens)
        addr = tokens{1}{1};
        l_addr = tokens{1}{2};
        tokens = regexp(l_addr, '(.*)[:](\d+)$', 'tokens');
        if ~isempty(tokens{1})
          l_host = tokens{1}{1};
          l_port = str2double(tokens{1}{2});
        else
          l_port = 0;
          if ~isempty(l_addr)
            l_host = l_addr;
          else
            l_host = 'INADDR_ANY';
          end
        end
      else
        l_port = 0;
        l_host = 'INADDR_ANY';
      end

      tokens = regexp(addr, '(.*)[:](\d+)$', 'tokens');
      host = tokens{1}{1};
      port = str2double(tokens{1}{2});
      
      if is_srvr
        obj.lcl_addr = host;
        obj.lcl_port = port;
        [obj.sock, err, obj.dst_addr, obj.dst_port] = ...
          openTCPSocketServer(obj.lcl_addr, obj.lcl_port, timeout);
      else
        obj.lcl_addr = l_host;
        obj.lcl_port = l_port;
        obj.dst_addr = host;
        obj.dst_port = port;
        [obj.sock, err] = openTCPSocketClient(host,port,l_host,l_port);
      end
      if ~isempty(err)
        error('Failed opening TCP Socket failed: %s', err);
      end
    end
    
    % Destructor. Close handle if it was opened by the constructor.
    function delete(obj)
      if ~isempty(obj.sock)
        closeTCPSocket(obj.sock);
      end
    end
    
    function str = getJSON(obj)
      timeout_str = sprintf('"cnct_timeout":%f, "recv_timeout":%f', ...
        obj.cnct_timeout, obj.recv_wait);
      if obj.cnct_timeout < 0
        srvr_str = '"sndr_is_srvr:true';
        addr_str = sprintf('"addr":"%s:%d::%s:%d"', ...
          obj.dst_addr, obj.dst_port, obj.lcl_addr,obj.lcl_port);
      else
        srvr_str = '"sndr_is_srvr":false';
        addr_str = sprintf('"addr":"%s:%d"', obj.lcl_addr,obj.lcl_port);
      end
      str = sprintf('{%s, %s, %s}', timeout_str, srvr_str, addr_str);
    end
    
    function df = getDefStruct(obj)
      df = struct('cnct_timeout', obj.cnct_timeout,...
        'recv_timeout', obj.recv_wait,...
        'sndr_is_srvr', (obj.cnct_timeout < 0));
      
      if df.sndr_is_srvr
        df.addr = sprintf('%s:%d',obj.lcl_addr,obj.lcl_port);
      else
        df.addr = sprintf('%s:%d::%s:%d',...
          obj.dst_addr, obj.dst_port, obj.lcl_addr,obj.lcl_port);
      end
    end
    
    function s = show_str(obj, fmt, params)
      if nargin < 3
        params = struct();
        if nargin <2
          fmt = struct();
        end
      end
      params.struct_marked = true;
      s = sprintf('%s<\n%s>', class(obj), ...
        show_str(obj.getDefStruct(), fmt, params));
    end
    
    function code = read(obj, cnt)
      cnt = double(cnt);
      code = zeros(cnt,1,'uint8');
      
      while cnt > 0
        avail = length(obj.buffer) - obj.buf_done;
        if avail >= cnt
          code(end-cnt+1:end) = obj.buffer(obj.buf_done+1:obj.buf_done+cnt);
          obj.buf_done = obj.buf_done + cnt;
          return;
        elseif avail > 0
          code(end-cnt+1:end-cnt+avail) = obj.buffer(obj.buf_done+1:end);
          cnt = cnt - avail;
          obj.buffer = [];
          obj.buf_done = 0;
        end
        
        [obj.buffer, err] = ...
          recvTCPSocket(obj.sock, max(cnt,obj.buf_size), obj.recv_wait);
        if ~isempty(err)
          code = err;
          return
        end
      end
    end
    
  end
  
  methods (Access=protected)
    % Read a sequence of bytes which contains a specified number of
    % bytes < 128 and return the seuqence and the indices.
    % Input:
    %    obj - this object
    %    max_cnt - (optional) maximal number of byte to read.
    %    nseq - number of bytes which are < 128
    % Output:
    %    buf - Output buffer (uint8)
    %          -1 if EOD was encountered before any byte was read. An
    %          error string if an error occurred or if EOD was
    %          encountered after some bytes were read or if max_cnt was
    %          exceeded.
    %    indcs - indices of the bytes < 128 (row vector of of nseq entries)
    %    cnt - Number of bytes read.
    function [buf, indcs, cnt] = readSeqs(obj, max_cnt, nseq)
      max_cnt = double(max_cnt);
      nseq = double(nseq);
      cnt = 0;
      if ~nseq
        indcs = [];
        buf = uint8([]);
        return
      end
      
      ns = 0;
      indcs = zeros(1,nseq);
      offset = obj.buf_done;
      
      while ns<nseq
        eds = find(obj.buffer(offset+1:end)<128, nseq-ns);
        if ~isempty(eds)
          cnt = cnt + eds(end);
          if cnt > max_cnt
            buf = 'Exceeded allowed number of bytes';
            return
          end
          ns1 = ns+length(eds);
          indcs(ns+1:ns1) = eds + (offset-obj.buf_done);
          ns = ns1;
          offset = offset + eds(end);
          continue;
        else
          % need to read
          ns_left = nseq - ns;
          nrd = max(min(4*ns_left, max_cnt-cnt), obj.buf_size);
          [bfr, err] = recvTCPSocket(obj.sock, nrd, obj.recv_wait);
          if ~isempty(err)
            buf = err;
            return
          end
          obj.buffer = [obj.buffer; bfr];
        end
      end
      buf = obj.buffer(obj.buf_done+1:obj.buf_done+indcs(end));
      obj.buf_done = obj.buf_done + indcs(end);
      if obj.buf_done == length(obj.buffer)
        obj.buffer = [];
        obj.buf_done = 0;
      elseif obj.buf_done > obj.buf_size
        obj.buffer = obj.buffer(obj.buf_done+1:end);
        obj.buf_done = 0;
      end
    end
  end
end

