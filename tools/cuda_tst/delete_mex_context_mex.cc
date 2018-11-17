/**
   \file

   delete mex context.
 */

#include <string.h>

#include "mex.h"
#include "mex_context.h"


//* This function deletes mex_context
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  (void) plhs;
  (void) nlhs;
  (void) prhs;
  (void) nrhs;

  mex_assert(nlhs==0 && nrhs==0,
	     ("delete_mex_contex_mtx:args",
	      "delete_mex_context should have no input and no output"));

  MexContext::deleteContext();
}

