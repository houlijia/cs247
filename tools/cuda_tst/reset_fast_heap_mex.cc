/**
   \file

   reset fast_heap
 */

#include "mex.h"
#include "fast_heap.h"
#include "mex_context.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mex_assert(nlhs == 0 && nrhs == 0,
	     ("reset_fast_heap_mex:InvalidInput",
	      "reset_fast_heap_mex should have no input and no output arguments"));

  reset_fast_heap();
}
