classdef CodeDestTCP < CodeDest
  % CodeDestTCP is an implementation of CodeDest as a TCP Socket, opened in
  % client mode (connect).
  
  properties (Constant)
    buf_size = 1024;
  end
  properties
    sock = [];   %socket
    cnct_timeout;
    linger_timeout = -1;
    lcl_addr   % local address to bind to (default: '')
    lcl_port   % Local port to bind to
    dst_addr   % IP address or name of receiver
    dst_port   % port of receiver
    datalen = 0;
    buffer = zeros(CodeDestTCP.buf_size,1, 'uint8');
    buf_len = 0;
  end
  
  methods
    function obj = CodeDestTCP(timeout, addr)
      % Constructor.
      % Input:
      %   timeout - (vector of length 1 or 2). timeout(1) specifies the
      %             listening time for connection. The values can be:
      %               -1: Open the TCP socket in client (connect) mode, so
      %                   timeout is not relevant.
      %               0: Open the TCP socket in server (listen) mode, and
      %                  wait indefinitely.
      %               >0: Open the TCP socket in server (listen) mode, and
      %                  wait timeout(1) seconds.
      %             timeout(2) if present, specify the time to linger when
      %             closing. -1 is use system default. A non-negative value
      %             is the number of seconds to wait (rounded up).
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
      %
      
      obj.cnct_timeout = timeout(1);
      is_srvr = (timeout(1) > 0);
      
      if length(timeout)>1 && timeout(2) >= 0
        obj.linger_timeout = ceil(timeout(2));
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
        error('Failed opening TCP Socket: %s', err);
      end
    end
    
    % Destructor. Close handle if it was opened by the constructor.
    function delete(obj)
      if ~isempty(obj.sock)
        if obj.buf_len > 0
          err = sendTCPSocket(obj.sock, obj.buffer(1:obj.buf_len));
          if ~isempty(err)
            error('sendTCPSocket failed: %s', err);
          end
        end
        err = closeTCPSocket(obj.sock, obj.linger_timeout);
        if ~isempty(err)
          error('closeTCPSocket failed: %s', err);
        end
      end
    end
    
    function str = getJSON(obj)
      timeout_str = sprintf('"cnct_timeout":%f, "linger_timeout:%f', ...
        obj.cnct_timeout, obj.linger_timeout);
      if obj.cnct_timeout < 0
        srvr_str = '"sndr_is_srvr":false';
        addr_str = sprintf('"addr":"%s:%d::%s:%d"', ...
          obj.dst_addr, obj.dst_port, obj.lcl_addr,obj.lcl_port);
      else
        srvr_str = '"sndr_is_srvr":true';
        addr_str = sprintf('"addr":"%s:%d"', obj.lcl_addr,obj.lcl_port);
      end
      str = sprintf('{%s, %s, %s}', timeout_str, srvr_str, addr_str);
    end
    
    function df = getDefStruct(obj)
      df = struct('cnct_timeout', obj.cnct_timeout,...
        'linger_timeout', obj.linger_timeout,...
        'sndr_is_srvr', (obj.cnct_timeout >= 0));
      
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
    
    function err = write(obj, code)
      code_len = length(code);
      len = code_len + obj.buf_len;
      if len >= obj.buf_size
        err = sendTCPSocket(obj.sock, [obj.buffer(1:obj.buf_len); code(:)]);
        if ~isempty(err)
          return
        else
          err = 0;
        end
        obj.buf_len = 0;
      else
        obj.buffer(obj.buf_len+1:obj.buf_len+code_len) = code(:);
        obj.buf_len = len;
        err = 0;
      end
      obj.datalen = obj.datalen + code_len;
    end
    
    function len = length(obj)
      len = obj.datalen;
    end
    
  end
  
end

