/**
   \file

   This MEX function computes the mean of a vector given in prhs[0] and
   returns the results in plhs[0]. The input can be on CPU or GPU and it can
   be double or single precision. The computation is done where the input is,
   but the result is always returned in CPU. If the input is double or single
   the computation is in double or single precision, and the result is double
   or single, respectively.

   If the input is empty the output is 0.
*/
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "CudaDevInfo.h"
#include "cuda_sum_mean_var.h"
#include "cc_sum_mean_var.h"
#include "mex_gpu_tools.h"
#include "fast_heap.h"
#include "mex_context.h"

static char const * const errId = "cuda_mean_mex:InvalidInput";

template<class Float>
static void calcGPU(const mxGPUArray *pr,
		    size_t n_vec,
		    mxArray *output)
{
  const Float *p_src_vec = (const Float*) mxGPUGetDataReadOnly(pr);
  Float *pmean = (Float *) mxGetData(output);

  if(n_vec > 0) {
    GenericHeapElement &pres = d_fast_heap->get(n_vec * sizeof(Float));
    Float *res = static_cast<Float*>(*pres);
 
    h_mean_vec(n_vec, p_src_vec, res);

    gpuErrChk(cudaMemcpy(pmean, res, sizeof(*res), cudaMemcpyDeviceToHost),
	      "cuda_mean_mex:memcpy","");
    pres.discard();
  }
  else 
    *pmean = 0;
}

template<class Float>
static void calcCPU(const mxArray *pr,
		    size_t n_vec,
		    mxArray *output)
{
  const Float *p_src_vec = (const Float*) mxGetData(pr);
  Float *pmean = (Float *)mxGetData(output);

  if(n_vec > 0)
    *pmean = c_mean_vec(n_vec, p_src_vec);
  else
    *pmean = 0;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxClassID class_id;
  size_t n_vec, n_dim, n_rows, n_cols;
  const mxGPUArray *pr;
  
  // Check correctness and get input information
  if(nlhs != 1 || nrhs != 1)
    mexErrMsgIdAndTxt(errId,
		      "cuda_mean_mex should have 1 input and 1 output arguments");

  int is_gpu = mxIsGPUArray(prhs[0]);

  if(is_gpu) {
    pr = mxGPUCreateFromMxArray(prhs[0]);
    if(mxGPUGetComplexity(pr) != mxREAL)
	mexErrMsgIdAndTxt(errId, "Input to cuda_mean_mex must be real");
    class_id = mxGPUGetClassID(pr);

    n_vec = (size_t) mxGPUGetNumberOfElements(pr);
    n_dim = (size_t) mxGPUGetNumberOfDimensions(pr);
    const mwSize *dims = mxGPUGetDimensions(pr);
    n_rows = dims[0];
    n_cols = dims[1];
    mxFree((void *)dims);
  }
  else {
    if(mxIsComplex(prhs[0]))
      mexErrMsgIdAndTxt(errId, "Input to cuda_mean_mex must be real");
    class_id = mxGetClassID(prhs[0]);

    n_vec = (size_t) mxGetNumberOfElements(prhs[0]);
    n_dim = (size_t) mxGetNumberOfDimensions(prhs[0]);
    n_rows = mxGetM(prhs[0]);
    n_cols = mxGetN(prhs[0]);
  }

  if(n_dim != 2 ||  (n_rows > 1 && n_cols > 1))
    mexErrMsgIdAndTxt(errId,
		      "cuda_mean_mex input should be a vector");
  if(class_id == mxSINGLE_CLASS || class_id == mxDOUBLE_CLASS)
    plhs[0] = mxCreateNumericMatrix(1, 1, class_id, mxREAL);
  else
    mexErrMsgIdAndTxt(errId,
		      "Input must be float (%d) of double (%d). Currnt type %d",
		      mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id);
    
  if(is_gpu) { 
    if(class_id == mxSINGLE_CLASS)
      calcGPU<float>(pr, n_vec, plhs[0]);
    else
      calcGPU<double>(pr, n_vec, plhs[0]);
    mxGPUDestroyGPUArray(pr);
  }
  else {
    if(class_id == mxSINGLE_CLASS)
      calcCPU<float>(prhs[0], n_vec, plhs[0]);
    else
      calcCPU<double>(prhs[0], n_vec, plhs[0]);
  }
}
