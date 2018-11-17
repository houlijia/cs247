/**
   \file

   A mex function for deleting C++ CodeElement object given a pointer to them (prhs[0]).
   If prhs[1] exists and is logically true, an array is deleted.

*/

#include <string.h>

#include "CodeElement.h"

static char const * const errId = "deleteCodeElement_mex:InvalidInput";

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  if(nrhs > 2 || nlhs > 0)
    mexErrMsgIdAndTxt(errId,
		      "deleteCodeElement_mex should have 1 or 2 input "
		      "and no output arguments");

  if(mxGetClassID(prhs[0]) != mxUINT8_CLASS)
    mexErrMsgIdAndTxt(errId, "First argument must be of class uint8");

  if(nrhs==2 && !mxIsLogicalScalar(prhs[1]))
    mexErrMsgIdAndTxt(errId, "Second argument must be a logical scalar");

  CodeElementPtr dev_ptr;
  memcpy(dev_ptr.b, mxGetData(prhs[0]), sizeof(dev_ptr));
  if(nrhs==2 && mxIsLogicalScalarTrue(prhs[1]))
    // Array case
    delete [] dev_ptr.ce;
  else
    delete dev_ptr.ce;
}

