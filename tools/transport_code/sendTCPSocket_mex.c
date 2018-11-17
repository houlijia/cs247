/**
  \file

  \brief Interface to the   MATLAB function:  sendTCPSocket_mex

  MATLAB usage:

  function [err] = sendTCPSocket_mex(sock, data)

  The function sends the array data over the socket.
  If an error occurs then, if err is specified err contains an error message;
  otherwise an exception is thrown. If successful and \c error is specified,
  \c err is empty.

  INPUT:
    sock - a socket object
    data - an vector array of uint8 items.

  OUTPUT (Note that at least the first output has to be specified):
    err - An error string. Empty string if no failure
 */
#include <stdio.h>
#include <math.h>
#include <mex.h>
#include <matrix.h>

#include "tcp_sock_io.h"
#include "tcp_sock_mex.h"

void mexFunction(int nlhs,
		 mxArray *plhs[],
		 int nrhs,
		 const mxArray *prhs[]
		 )
{
  const char *err_str;
  socket_t sock;
  size_t datalen, m, n;
  const void * data;

  if(nrhs != 2)
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "Expected 2, found %d", POS_PRNT, nrhs);

  if(!mxIsSocket(prhs[0]))
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "First argument is not a socket", POS_PRNT);

  if(!(mxIsUint8(prhs[1]) && mxGetNumberOfDimensions(prhs[1])==2 &&
       (mxGetM(prhs[1])<=1 || mxGetN(prhs[1])<=1)))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT
		      "Second argument is not a vector of UInt8", POS_PRNT);
 
  if(nlhs > 1)
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "(Expected 0 or 1 output arguments, found %d",
		      POS_PRNT, nlhs);

  
  mxGetSocket(prhs[0], sock);
  m = mxGetM(prhs[1]);
  n = mxGetN(prhs[1]);
  datalen = (m>n)? m: n;
  data = mxGetData(prhs[1]);

  if(sendTCPSocket(sock, datalen, data, &err_str) != 0) {
    if(nlhs > 0)
      plhs[0] = mxCreateString(err_str);
    else
      mexErrMsgIdAndTxt(ERR_ID("Send"), POS_FMT "Send failed: %s", POS_PRNT, err_str);
  }
  else if(nlhs > 0)
    plhs[0] = mxCreateString("");
}

