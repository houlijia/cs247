/**
  \file

  \brief Interface to the   MATLAB function:  openTCPSocketServer_mex

  MATLAB usage:

  function [sock, err, host, port, err] = 
        openTCPSocketServer_mex(lcl_host, lcl_port, timeout)
  The function opens a TCP socket in server mode (i.e. listening) and
  waits until one client connects to the server. Then it returns and
  closes the listening socket. If an error occurs then, if \c err
  is specified \c err contains an error message; otherwise an
  exception is thrown. If successful and \c err is specified,
  \c err is empty.

  INPUT:
    lcl_host - a string specifying the local interface to bind to. Can
               be a name or an IP address in dotted decimal notation
               or in the form 0{x|X}\<hex no\> (host order). If empty,
               INADDR_ANY is used.
    lcl_port - A port number to bind to.
    timeout - (optional) timeout in seconds, 0 = indefinite (default=0).

  OUTPUT (Note that at least the first output has to be specified):
    sock - An uint8 array representing the socket. Empty of open failed.
    err - An error string. Empty string if no failure
    host - a string containing the IP address of the client, in
      dotted decimal notation.
    port - port number of the client (host order) - a uint16 number.
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
  char *lcl_host, host[256];
  unsigned long lcl_port, port;
  double timeout = 0.;
  socket_t sock;
  const char *err_str;

  if(nrhs<2 || nrhs>3)
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "Expected 2 or 3 input arguments, found %d", POS_PRNT, nrhs);

  if(!mxIsString(prhs[0]))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "First argument is not a string", POS_PRNT);

  lcl_port = mxGetPort(prhs[1], 1, "Second argument");

  if(nrhs == 3) {
    timeout = mxGetScalar(prhs[2]);
    if(timeout < 0)
      mexErrMsgIdAndTxt(ERR_ID("Args"),
			POS_FMT "Timeout cannot be negative", POS_PRNT);
  }

  if(nlhs <= 0 || nlhs > 4) {
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "(Expected 1 to 4 output arguments, found %d",
		      POS_PRNT, nlhs);
  }

  if(mxIsEmpty(prhs[0]))
    lcl_host  = mxAllocINADDR_ANYString();
  else
    lcl_host = mxAllocGetString(prhs[0]);

  sock = openTCPSocketServer(lcl_host, lcl_port, sizeof(host), host, &port, timeout, &err_str);

  mxFree(lcl_host);

  if(err_str != NULL) {
    if(nlhs >= 2) {
      plhs[0] = mxCreateNumericMatrix(0,0,mxUINT8_CLASS, mxREAL);
      plhs[1] = mxCreateString(err_str);
      if (nlhs >= 3) {
	plhs[2] = mxCreateString("");
	if(nlhs == 4) {
	  uint16_T z=0;
	  plhs[3] = mxCreateNumericScalar(&z, mxUINT16_CLASS, sizeof(z));
	}
      }
      return;
    }
    else
      mexErrMsgIdAndTxt(ERR_ID("Open"), POS_FMT "Open failed: %s", POS_PRNT, err_str);
  }

  plhs[0] = sockToMxArray(sock);

  if(nlhs > 1) {
    plhs[1] = mxCreateString("");
    if(nlhs > 2) {
      plhs[2] = mxCreateString(host);
      if(nlhs > 3)
	plhs[3] = mxCreateNumericScalar(&port, mxUINT16_CLASS, sizeof(port));
    }
  }
}
