/**
   \file

   Check is Mex SW is build for GCC
 */

#include "mex.h"

#include "mex_assert.h"

//* this function returns true if compiled with NVCC and false otherwise
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  (void) prhs;
  (void) nlhs;
  (void) nrhs;

  mex_assert(nlhs==1 && nrhs==0,
	     ("mexGPU__mex:args",
	      "mexGPU_mex.cc should have no inputs and one output"));

#ifdef __NVCC__
  plhs[0] = mxCreateLogicalScalar(true);
#else
  plhs[0] = mxCreateLogicalScalar(false);
#endif

}
