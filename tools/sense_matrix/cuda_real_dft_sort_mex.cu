/**
   \file 

   MEX functions to perform quantization. 
 */

#include <string.h>
#include <math.h>

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mex_tools.h"
#include "mex_gpu_tools.h"
#include "mex_assert.h"
#include "CudaDevInfo.h"
#include "cc_real_dft_sort.h"
#include "cuda_real_dft_sort.h"
#include "mex_context.h"

static char const * const errId = "cuda_real_dft_sort_mex:InvalidInput";

/**
   This MEX function performs real DFT sorting into a compact form. 
   Input:
     prhs[0] - (\c cf) A matrix of complex DFT coefficients (assuming that the
               DFT was along columns). Can be single or double precision
   Output:
     plhs[0] - (cmpct) A real matrix of the same size and class as \c cf,
               containing the DFT coefficients in a comact form.
   Input can be either on GPU or on CPU. Outputs are in the same place
   as the input.
 */

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxClassID class_id;
  size_t N;			// DFT order
  size_t ncl;			// number of columns
  const mxGPUArray *pr;

  // Check correctness and get input information
  mex_assert(nlhs==1 && nrhs==1,
	     (errId,"Should have 1 input and 1 output arguments"));

  int is_gpu = mxIsGPUArray(prhs[0]);
  
  if(is_gpu) {
    pr = mxGPUCreateFromMxArray(prhs[0]);
    class_id = mxGPUGetClassID(pr);
    mwSize const *dims = mxGPUGetDimensions(pr);
    N = dims[0];
    ncl = dims[1];
    mxFree((void *)dims);

    mex_assert(((class_id == mxSINGLE_CLASS || class_id == mxDOUBLE_CLASS) &&
		mxGPUGetComplexity(pr) != mxREAL &&
		mxGPUGetNumberOfDimensions(pr) == 2 &&
		N>0 &&
		ncl>0),
	       (errId,"First argument a non-empty, complex, single or double float vector or matrix"));
  }
  else {
    class_id = mxGetClassID(prhs[0]);
    N = mxGetM(prhs[0]);
    ncl = mxGetN(prhs[0]);

    mex_assert(((class_id == mxSINGLE_CLASS || class_id == mxDOUBLE_CLASS) &&
		mxIsComplex(prhs[0]) &&
		mxGetNumberOfDimensions(prhs[0]) == 2 &&
		N>0 &&
		ncl>0),
	       (errId,"First argument a non-empty, complex, single or double float vector or matrix"));
  }

  
 if(is_gpu) {
    mxGPUArray *cmpct = create_GPUMatrix(N, ncl, class_id);

    if(class_id == mxSINGLE_CLASS)
      h_real_dft_sort(N, ncl, (const float *)mxGPUGetDataReadOnly(pr),
		      (float *)mxGPUGetData(cmpct));
    else
      h_real_dft_sort(N, ncl, (const double *)mxGPUGetDataReadOnly(pr),
		      (double *)mxGPUGetData(cmpct));

    plhs[0] = mxGPUCreateMxArrayOnGPU(cmpct);
    mxGPUDestroyGPUArray(cmpct);
    mxGPUDestroyGPUArray(pr);
  }
  else {
    plhs[0] = mxCreateNumericMatrix(N, ncl, class_id, mxREAL);

    if(class_id == mxSINGLE_CLASS)
      c_real_dft_sort(N, ncl, (const float *)mxGetData(prhs[0]),
		      (const float *)mxGetImagData(prhs[0]),
		      (float *)mxGetData(plhs[0]));
    else
      c_real_dft_sort(N, ncl, (const double *)mxGetData(prhs[0]),
		      (const double *)mxGetImagData(prhs[0]),
		      (double *)mxGetData(plhs[0]));
  }
}
