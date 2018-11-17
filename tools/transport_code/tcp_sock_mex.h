#ifndef TCP_SOCK_MEX_H
#define TCP_SOCK_MEX_H

#include <string.h>
#include <mex.h>

#include "tcp_sock_io.h"
#include "mex_tools.h"

#define mxIsSocket(x) \
  (mxIsUint8(x) && mxGetNumberOfDimensions(x)==2 && \
   mxGetM(x)==sizeof(socket_t) && mxGetN(x)==1)

#define mxGetSocket(x,sock) memcpy(&(sock), mxGetData(x), sizeof(socket_t))

#define ERR_ID(str) ("TCPSocket:" str)

#define POS_PRNT  mexFunctionName(),__FILE__,__LINE__
#define POS_FMT "%s:%s:%d "

/** Allocate and copy the string in mxstr to a C string */
char *mxAllocGetString(const mxArray *mxstr);

char *mxAllocINADDR_ANYString(void);

/** Check that x i port numbe with value (host order) >= minval
    and return the port number. Throw an exception if an error occurs*/
unsigned long mxGetPort(const mxArray *mxarr,
		       unsigned long minval,
		       const char *dscr /**< Input description for error message */
		       );

/** Create and return a pointer to a mxArray object repreesnting a
    scalar of class \c classid, which occupies \c size bytes, and copy
    to it the value pointed to by \c pval
*/
mxArray *mxCreateNumericScalar(const void *pval, mxClassID classid, size_t size);

mxArray *sockToMxArray(socket_t sock); 

#endif	/* TCP_SOCK_MEX_H */
