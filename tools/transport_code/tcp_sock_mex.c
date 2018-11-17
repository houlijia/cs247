#include <stddef.h>
#include <math.h>

#include "tcp_sock_mex.h"
#include "tcp_sock_io.h"

char *mxAllocGetString(const mxArray *mxstr)
{
  size_t len;
  char *str;

  len = mxGetN(mxstr)*sizeof(mxChar)+1;
  str = mxMalloc(len);
  mxGetString(mxstr, str, (mwSize)len);
  return str;
  
}

char *mxAllocINADDR_ANYString(void)
{
  static char addr_str[20];
  static size_t len=0;
  char * str;
  
  if(!len) {
    ip_to_str(INADDR_ANY, addr_str);
    len = strlen(addr_str);
  }

  str = mxMalloc(len+1);
  strcpy(str, addr_str);
  return str;
}

/** Check that x i port numbe with value (host order) >= minval
    and return the port number. Throw an exception if an error occurs*/
unsigned long mxGetPort(const mxArray *mxarr,
		       unsigned long minval,
		       const char *dscr /**< Input description for error message */
		       )
{
  double dbl_port;

  if(!mxIsNumeric(mxarr) || !mxIsScalar(mxarr))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT "%s is not scalar numric", POS_PRNT, dscr);
  
  dbl_port = mxGetScalar(mxarr);

  if(dbl_port != floor(dbl_port) || dbl_port < minval || dbl_port > (double)((1L<<16)-1))
    mexErrMsgIdAndTxt(ERR_ID("Args"), POS_FMT
		      "%s is not a 16-bit integer not less than %lu",
		      POS_PRNT, dscr, minval);
  return (unsigned long) dbl_port;
}

mxArray *mxCreateNumericScalar(const void *pval, mxClassID classid, size_t size)
{
  mxArray *mxr;

  mxr = mxCreateNumericMatrix(1,1,classid, mxREAL);
  memcpy(mxGetData(mxr),pval, size);

  return mxr;
}

mxArray *sockToMxArray(socket_t sock)
{
  mxArray *mxr;

  mxr = mxCreateNumericMatrix(sizeof(sock),1, mxUINT8_CLASS, mxREAL);
  memcpy(mxGetData(mxr), &sock, sizeof(sock));
  return mxr;
}


