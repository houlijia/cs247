/**
   This function takes a scalar prhs[0] and a vector prhs[1] and returns in
   plhs[0] the vector resulting from subtracting the scalar from the vector
   and then taking square of each element. Computation is done on the GPU, and
   result is returned in the GPU, if the inputs are on the GPU. Comptutation
   is in single or double precision float, according to the precision of thye input.
 */
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "CudaDevInfo.h"
#include "cuda_sub_sqr.h"
#include "mex_context.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  char const * const errId = "cuda_sub_sqr_mex:InvalidInput";
  int k;
  mxClassID class_id[2];
  size_t n_vec, n0;
  const mxGPUArray *pr[2];
  const void *p_sclr, *p_src_vec;
  void *p_dst_vec;
  
  // Check correctness and get input information
  if(nlhs != 1 || nrhs != 2)
    mexErrMsgIdAndTxt(errId,
		      "cuda_sub_sqr_mex should have 2 inputs and 1 output arguments");

  if(!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[1]))
      mexErrMsgIdAndTxt(errId,
                        "Inputs to cuda_sub_sqr_gmex should GPUArray class");
  for(k=0; k<2; k++) {
      pr[k] = mxGPUCreateFromMxArray(prhs[k]);
      if(mxGPUGetComplexity(pr[k]) != mxREAL)
          mexErrMsgIdAndTxt(errId, "Inputs to cuda_sub_sqr_mex must be real");
      class_id[k] = mxGPUGetClassID(pr[k]);
  }
  n0 = (size_t) mxGPUGetNumberOfElements(pr[0]);
  if(class_id[0] != class_id[1])
      mexErrMsgIdAndTxt(errId, "Inputs of the same type");

  if(class_id[0] != mxDOUBLE_CLASS && class_id[0] != mxSINGLE_CLASS)
      mexErrMsgIdAndTxt(errId,
                        "Inputs must be float (%d) of double (%d). Currnt type %d",
                        mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id[0]);

  if(n0 != 1)
      mexErrMsgIdAndTxt(errId, "First input arguments must be a scalar");

  // Do the actual processing
  p_sclr = mxGPUGetDataReadOnly(pr[0]);
  p_src_vec = mxGPUGetDataReadOnly(pr[1]);
  n_vec = (size_t) mxGPUGetNumberOfElements(pr[1]);
  const mwSize *dims =  mxGPUGetDimensions(pr[1]);

  mxGPUArray *result = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(pr[1]),
                                           dims,
                                           class_id[0], 
                                           mxREAL,
                                           MX_GPU_DO_NOT_INITIALIZE);
  p_dst_vec = mxGPUGetData(result);
  switch(class_id[0]) {
  case mxDOUBLE_CLASS:
      h_sub_sqr<double>((const double *)p_sclr, (const double *)p_src_vec, n_vec,
                        (double *)p_dst_vec);
      break;
  case mxSINGLE_CLASS:
      h_sub_sqr<float>((const float *)p_sclr, (const float *)p_src_vec, n_vec,
                       (float *)p_dst_vec);
      break;
  }
    
  plhs[0] = mxGPUCreateMxArrayOnGPU(result);

  mxFree((void *)dims);
  mxGPUDestroyGPUArray(result);
  mxGPUDestroyGPUArray(pr[1]);
  mxGPUDestroyGPUArray(pr[0]);
}

