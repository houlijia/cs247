/**
  \file

  \brief Interface to the   MATLAB function:  sendTCPSocket_mex

  MATLAB usage:

  function [data, err] = recvTCPSocket_mex(sock, nbyte, timeout)

  The function receives up to nbyte bytes from the socket and returns
  them in \c data. 
  If an error occurs then, if err is specified err contains an error message,
  otherwise an exception is thrown. If successful and \c error is specified,
  \c err is empty.

  INPUT:
    sock - a socket object
    nbyte - a non-negative integer specifying the requested data length.
    timeout - (optional) duration to wait (sec.) for data. 0=wait indefinitely.
              default: 0 

  OUTPUT (Note that at least the first output has to be specified):
    data - the read bytes, a uint8 vector of length <= \c nbyte. An empty string
           indicates timeout, or remote side has closed, or an error occured.
    err - An error string if an error occured, -1 if remote side closed
          gracefully Empty string otherwise
 */
#include <stdio.h>
#include <stddef.h>
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
  double dbl_nbyte;
  size_t nbyte;
  double timeout = 0;
  SSIZE_T datalen;
  mxArray *mxr;
  void * data;

  if(nrhs<2 || nrhs>3)
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "Expected 2 or 3 arguments, found %d", POS_PRNT, nrhs);

  if(!mxIsSocket(prhs[0]))
    mexErrMsgIdAndTxt(ERR_ID("Args"),
		      POS_FMT "First argument is not a socket", POS_PRNT);
  
  mxGetSocket(prhs[0], sock);

  if(!mxIsNumeric(prhs[1]) || ! mxIsScalar(prhs[1]))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT
		      "Second argument is not a numeric scalar", POS_PRNT);

  dbl_nbyte = mxGetScalar(prhs[1]);
  if(dbl_nbyte < 0 || dbl_nbyte != floor(dbl_nbyte))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT
		      "Second argument is not a non-negative integer", POS_PRNT);
  nbyte = (size_t) dbl_nbyte;

  if(nrhs == 3) {
    if(!mxIsNumeric(prhs[2]) || ! mxIsScalar(prhs[2]))
      mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT
			"Third argument is not a numeric scalar", POS_PRNT);
    timeout = mxGetScalar(prhs[2]);
  }

  if(nlhs > 2)
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "(Expected up to 2 output arguments, found %d",
		      POS_PRNT, nlhs);

  mxr = mxCreateNumericMatrix(nbyte, 1, mxUINT8_CLASS, mxREAL);
  data = mxGetData(mxr);

  datalen = recvTCPSocket(sock, nbyte, data, timeout, &err_str);
  if(datalen == SOCKET_ERROR) {
    mxSetM(mxr, 0);
    mxSetN(mxr, 0);
  }
  else if((size_t)datalen < nbyte) {
    mxSetM(mxr, (mwSize) datalen);
    if(datalen == 0)
      mxSetN(mxr, 0);
  }

  if(nlhs > 0) {
    plhs[0] = mxr;

    if(nlhs > 1) {
      if(err_str == NULL) {
	if(datalen == SOCKET_ERROR)
	  plhs[1] = mxCreateDoubleScalar(-1.);
	else if (datalen==0 && nbyte>0)
	  plhs[1] = mxCreateString("Timed out waiting for data");
	else
	  plhs[1] = mxCreateString("");
      }
      else
	plhs[1] = mxCreateString(err_str);
    }
  }
  else 
    mxDestroyArray(mxr);
}
	
 
    
  

