function [ varargout ] = recvTCPSocket( varargin )
  %recvTCPSocket read from socket
  %
  %   Interface to the   MATLAB function:  sendTCPSocket_mex
  %
  %   MATLAB usage:
  %
  %   function [data, err] = recvTCPSocket(sock, nbyte, timeout)
  %
  %   The function receives up to nbyte bytes from the socket and returns
  %   them in \c data.
  %   If an error occurs then, if err is specified err contains an error message,
  %   otherwise an exception is thrown. If successful and \c error is specified,
  %   \c err is empty.
  %
  %   INPUT:
  %     sock - a socket object
  %     nbyte - a non-negative integer specifying the requested data length.
  %   timeout - (optional) duration to wait (sec.) for data. 0=wait indefinitely.
  %             default: 0 
  %
  %   OUTPUT (Note that at least the first output has to be specified):
  %     data - the read bytes, a uint8 vector of length <= \c nbyte.
  %     err - An error string. Empty string if no failure
  
  narginchk(2,3)
  nargoutchk(0,2)
  
  varargout = cell(1,nargout);
  [varargout{:}] = recvTCPSocket_mex(varargin{:});
  
end

