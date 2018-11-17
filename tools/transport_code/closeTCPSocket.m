function [ varargout ] = closeTCPSocket( varargin )
  % closeTCPSocket closes a TCP socket
  %
  % This function is an interface to the mex function closeTCPSocket_mex()
  % MATLAB Usage:
  %   function [err] = closeTCPSocket_mex(sock)
  %
  % The function closes a TCP socket. If err is specified: If
  % closing is successful err is an empty string, otherwise it
  % contains an error message. If no output arguments are specified and
  % an error occurs, an exception is thrown.
  %
  % INPUT:
  %   sock - socket (opaque object)
  %   lngr - (optional) linger timeout. negative = use system default (this is the
  %          defult if the argument is not specified). otherwise number of seconds
  %          to wait for other side to complete closing (rounded up to integer).
  % OUTPUT:
  %   err - An error string. Empty if no error.
  
  narginchk(1,2)
  nargoutchk(0,1)
  
  varargout = cell(1,nargout);
  [varargout{:}] = closeTCPSocket_mex(varargin{:});

end

