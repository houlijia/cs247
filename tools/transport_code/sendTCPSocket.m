function [ varargout ] = sendTCPSocket( varargin )
  % sendTCPSocket Send bytes over a TCP socket
  %
%   Interface to the   MATLAB function:  sendTCPSocket_mex
% 
%   MATLAB usage:
% 
%   function [err] = sendTCPSocket_mex(sock, data)
% 
%   The function sends the array data over the socket.
%   If an error occurs then, if err is specified err contains an error message;
%   otherwise an exception is thrown. If successful and \c error is specified,
%   \c err is empty.
% 
%   INPUT:
%     sock - a socket object
%     data - an vector array of uint8 items.
% 
%   OUTPUT (Note that at least the first output has to be specified):
%     err - An error string. Empty string if no failure

  narginchk(2,2)
  nargoutchk(0,1)
  
  varargout = cell(1,nargout);
  [varargout{:}] = sendTCPSocket_mex(varargin{:});

end

