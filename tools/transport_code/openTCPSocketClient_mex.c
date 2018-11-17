/**
  \file

  \brief Interface to the   MATLAB function:  openTCPSocketClient_mex

  MATLAB usage:

  function [sock, err, err_msg] = 
        openTCPSocketClient_mex(host, port, lcl_host, lcl_port)
  The function opens a TCP socket in client mode and connects to th a server. 
  If an error occurs then, if err is specified err contains an error message;
  otherwise an exception is thrown. If successful and \c error is specified,
  \c err is empty.

  INPUT:
    host - a string containing the IP address of the server, Can
               be a name or an IP address in dotted decimal notation
               or in the form 0{x|X}\<hex no\> (host order).
    port - port number of the server (host order) - a uint16 number.
    lcl_host - (optional) a string specifying the local interface to bind to. Can
               be a name or an IP address in dotted decimal notation
               or in the form 0{x|X}\<hex no\> (host order). If not present or
               empty, INADDR_ANY is used.
    lcl_port - (optional) A port number to bind to. If not present, 0 is used.

  OUTPUT (Note that at least the first output has to be specified):
    sock - An uint8 array representing the socket. Empty of open failed.
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
  char *lcl_host, *host;
  unsigned long lcl_port, port;
  socket_t sock;
  const char *err_str;

  if(nrhs < 2 || nrhs > 4)
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "Expected 2 to 4 input arguments, found %d", POS_PRNT, nrhs);

  if(!mxIsString(prhs[0]) || mxIsEmpty(prhs[0]))
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "First argument is not a non-empty string", POS_PRNT);

  host = mxAllocGetString(prhs[0]);
  port = mxGetPort(prhs[1], 1, "Second argument");

  if(nrhs > 2) {
    if(!mxIsString(prhs[2]))
      mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "Third argument is not a string", POS_PRNT);

    if(mxIsEmpty(prhs[2]))
      lcl_host  = mxAllocINADDR_ANYString();
    else
       lcl_host = mxAllocGetString(prhs[2]);
  }
  else
    lcl_host  = mxAllocINADDR_ANYString();

  if(nrhs > 3)
    lcl_port = mxGetPort(prhs[3], 0, "Fourth argument");
  else
    lcl_port = 0;

  if(nlhs <= 0 || nlhs > 2) {
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "(Expected 1 or 2 output arguments, found %d",
		      POS_PRNT, nlhs);
  }

  sock = openTCPSocketClient(host, port, lcl_host, lcl_port, &err_str);

  mxFree(host);
  mxFree(lcl_host);

  if(err_str != NULL) {
    if(nlhs >= 2) {
      plhs[0] = mxCreateNumericMatrix(0,0,mxUINT8_CLASS, mxREAL);
      plhs[1] = mxCreateString(err_str);
      return;
    }
    else
      mexErrMsgIdAndTxt(ERR_ID("Open"), POS_FMT "Open failed: %s", POS_PRNT, err_str);
  }

  plhs[0] = sockToMxArray(sock);

  if(nlhs > 1) {
    plhs[1] = mxCreateString("");
  }
}
