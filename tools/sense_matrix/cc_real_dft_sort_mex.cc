/**
   \file 

   MEX functions to perform quantization. 
 */

#include <string.h>
#include <math.h>

#include "mex.h"
#include "mex_tools.h"
#include "cc_real_dft_sort.h"

static char const * const errId = "cc_real_dft_sort_mex:InvalidInput";

/**
   This MEX functions performs real DFT sorting into a compact form. 
   Input:
     prhs[0] - (\c cf) A matrix of complex DFT coefficients (assuming that the
               DFT was along columns). Can be single or double precision
   Output:
     plhs[0] - (cmpct) A real matrix of the same size and class as \c cf,
               containing the DFT coefficients in a comact form.
 */

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxClassID class_id;
  bool is_cmplx;
  mwSize ndim;
  size_t N;			// DFT order
  size_t ncl;			// number of columns

  // Check correctness and get input information
  if(nlhs!=1 ||  nrhs!=1)
    mexErrMsgIdAndTxt(errId,
		      "Should have 1 input and 1 output arguments");

  class_id = mxGetClassID(prhs[0]);
  is_cmplx = mxIsComplex(prhs[0]);
  ndim = mxGetNumberOfDimensions(prhs[0]);
  N = mxGetM(prhs[0]);
  ncl = mxGetN(prhs[0]);

  if((class_id != mxSINGLE_CLASS && class_id != mxDOUBLE_CLASS) ||
     !is_cmplx || ndim != 2 || N==0 || ncl==0)
    mexErrMsgIdAndTxt(errId,
		      "First argument a non-empty, complex, single of double float "
		      "vector or matrix");

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
