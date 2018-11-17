/**
   \file

   initialize the mex context.
 */

#include <string.h>

#include "mex.h"
#include "mex_context.h"


//* This function create mex_context and returns a pointer to it
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  (void) prhs;
  (void) nlhs;
  (void) nrhs;

  mex_assert(nlhs==1 && nrhs==0,
	     ("init_mex_context_mex:args",
	      "init_mex_context should have no inputs and one output"));

  //  Using resetContext() instead of getContext() becasue of some apparent bug in the mex, where
  //  the static class member MexContext::mex_context does not get initialized to NULL.
  const MexContext * pcntxt = MexContext::resetContext();

  // Create output
  plhs[0] = mxCreateNumericMatrix(1, sizeof(pcntxt), mxUINT8_CLASS, mxREAL);
  mex_assert((plhs[0] != NULL), ("init_mex_contex_mtx:alloc", 
				 "Failed to allocate %lu uint8 for output",
				 (unsigned long) sizeof(pcntxt)));

  memcpy(mxGetData(plhs[0]), &pcntxt, sizeof(pcntxt));
}  
