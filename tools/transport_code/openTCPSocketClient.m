function [ varargout ] = openTCPSocketClient( varargin )
  % openTCPSocketClient Open a TCP socket in client (connect) mode.
  %
  %  Interface to the   MATLAB function:  openTCPSocketClient_mex
  %
  %   MATLAB usage:
  %
  %   function [sock, err] =
  %         openTCPSocketClient(host, port, lcl_host, lcl_port)
  %
  %   The function opens a TCP socket in client mode and connects to th a server.
  %   If an error occurs then, if err is specified err contains an error message;
  %   otherwise an exception is thrown. If successful and \c error is specified,
  %   \c err is empty.
  %
  %   INPUT:
  %     host - a string containing the IP address of the server, Can
  %                be a name or an IP address in dotted decimal notation
  %                or in the form 0{x|X}\<hex no\> (host order).
  %     port - port number of the server (host order) - a uint16 number.
  %     lcl_host - (optional) a string specifying the local interface to bind to. Can
  %                be a name or an IP address in dotted decimal notation
  %                or in the form 0{x|X}\<hex no\> (host order). If not present or
  %                empty, INADDR_ANY is used.
  %     lcl_port - (optional) A port number to bind to. If not present, 0 is used.
  %
  %   OUTPUT (Note that at least the first output has to be specified):
  %     sock - An uint8 array representing the socket. Empty of open failed.
  %     err - An error string. Empty string if no failure
  
  narginchk(1,4)
  nargoutchk(2,2)
  
  varargout = cell(1,nargout);
  [varargout{:}] = openTCPSocketClient_mex(varargin{:});

end

