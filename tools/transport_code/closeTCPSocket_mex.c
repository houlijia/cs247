/**
  \file

  \brief Interface to the   MATLAB function:  closeTCPSocket_mex

  MATLAB usage:

  function [err] = closeTCPSocket_mex(sock)

  The function closes a TCP socket. If err is specified: If
  closing is successful err is an empty string, otherwise it
  contains an error message. If no output arguments are specified and
  an error occurs, an exception is thrown.

  INPUT:
    sock - an opaque socket object
    lngr - (optional) linger timeout. negative = use system default (this is the
           defult if the argument is not specified). otherwise number of seconds
           to wait for other side to complete closing (rounded up to integer).

  OUTPUT:
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
  socket_t sock;
  const char *err_str;
  int lngr = -1;
  double val;


  if(nrhs < 1 || nrhs > 2)
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "Expected 1 or 2 input arguments, found %d", POS_PRNT, nrhs);
  if(!mxIsSocket(prhs[0]))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "First argument is not a socket", POS_PRNT);
  if(nrhs == 2) {
    if(!mxIsScalar(prhs[1]) && mxIsNumeric(prhs[1]))
      mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "Second argument is not a numeric scalar", POS_PRNT);

    val = mxGetScalar(prhs[1]);
    lngr = (val<0)? -1: (int)ceil(val);
  }

  if(nlhs > 1)
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "(Expected 0 or 1 output arguments, found %d",
		      POS_PRNT, nlhs);

  mxGetSocket(prhs[0], sock);

  if(closeTCPSocket(sock, lngr, &err_str)) {
    if(nlhs > 0)
      plhs[0] = mxCreateString(err_str);
    else
      mexErrMsgIdAndTxt(ERR_ID("Close"), POS_FMT "Close failed: %s", POS_PRNT, err_str);
  }
  else if(nlhs > 0)
    plhs[0] = mxCreateString("");
}
