/**
  \file

  \brief A MATLAB function undefSocket_mex(), which returns a socket which
  corresponds to an undefined socket.

  In Unix, this is socket number -1. In Windows it is whatever
  INVALID_SOCKET stands for. In both cases, it is returned as an array
  of uint8.

  MATLAB usage

  function undef_sock = undefSocket_mex()

  INPUT:
    None
  OUTPUT:
    sock - an uint8 array representing an undefined socket.
  
*/

#include <string.h>
#include <stdio.h>
#include <mex.h>

#include "tcp_sock_io.h"

void mexFunction(int nlhs,
		mxArray *plhs[],
		int nrhs,
		const mxArray *prhs[]
		)
{
  char err_msg[256];
  const size_t SOCKLEN = sizeof(socket_t);
  union {
    socket_t s;
    unsigned char c[1];
  } sock;
  if(nrhs > 0) {
    sprintf(err_msg,"Expected 0 input arguments, found %d", nrhs);
    mexErrMsgIdAndTxt(mexFunctionName(), err_msg);
  }

  if (nlhs == 0)
    return;
  else if(nlhs > 1) {
    sprintf(err_msg,"Expected 1 output arguments, found %d", nlhs);
    mexErrMsgIdAndTxt(mexFunctionName(), err_msg);
  }

  sock.s = SOCKET_T_UNDEF;

  plhs[0] = mxCreateNumericMatrix(SOCKLEN, 1, mxUINT8_CLASS, mxREAL);
  if(plhs[0] == NULL)
    mexErrMsgIdAndTxt(mexFunctionName(), "Failed to create socket space");

  memcpy(mxGetData(plhs[0]), sock.c, SOCKLEN);
}
