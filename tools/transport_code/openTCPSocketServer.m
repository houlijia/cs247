function [ varargout ] = openTCPSocketServer( varargin )
  % openTCPSocketServer Open a TCP socket in server (listening) mode.
  %
  % Interface to the mex function:  openTCPSocketServer_mex
  %
  % MATLAB usage:
  %   function [sock, err, host, port, err_msg] = ...
  %         openTCPSocketServer(lcl_host, lcl_port, timeout)
  % The function opens a TCP socket in server mode (i.e. listening) and
  % waits until one client connects to the server. Then it returns and
  % closes the listening socket. If an error occurs then, if \c err is
  % specified \c err contains an error message; otherwise an exception
  % is thrown. If successful and \c error is specified,\c err is empty.
  %
  % INPUT:
  %   lcl_host - a string specifying the local interface to bind to. Can
  %              be a name or an IP address in dotted decimal notation
  %              or in the form 0{x|X}\<hex no\> (host order). If empty,
  %              INADDR_ANY is used.
  %   lcl_port - A port number to bind to.
  %   timeout  - (optional) timeout in seconds, 0 = indefinite (default=0).
  %
  % OUTPUT (Note that at least the first output has to be specified):
  %   sock - An uint8 array representing the socket. Empty of open failed.
  %   err - An error string. Empty string if no failure
  %   host - a string containing the IP address of the client, in
  %     dotted decimal notation.
  %   port - port number of the client (host order) - a uint16 number.
    
  narginchk(2,3)
  nargoutchk(1,4)
  
  varargout = cell(1,nargout);
  [varargout{:}] = openTCPSocketServer_mex(varargin{:});
end

