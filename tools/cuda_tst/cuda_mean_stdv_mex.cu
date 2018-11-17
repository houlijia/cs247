
/**
   \file

   This MEX function computes the mean and standard deviation of a vector
   given in prhs[0] and returns the results in CPU. plhs[0] is the mean and
   plhs[1] is the standard variation. The input can be on CPU or GPU and it
   can be double or single precision. The computation is done where the input
   is, but the results are always returned in CPU. If the input is double or
   single the computation is in double or single precision, and the results
   are double or single, respectively.

   If the input is empty both outputs are 0. If it has one entry, only the mean is
   computed and the standard deviation is set to zero.
*/
#include "mex.h"
#include "mex_gpu_tools.h"
#include "CudaDevInfo.h"
#include "cuda_sum_mean_var.h"
#include "cc_sum_mean_var.h"
#include "mex_assert.h"
#include "fast_heap.h"
#include "mex_context.h"
#include "timers.h"

template<class Float>
static void calcGPU(const mxGPUArray *pr,
		    size_t n_vec,
		    mxArray *output[2]
		    )
{
  const Float *p_src_vec = (const Float*)mxGPUGetDataReadOnly(pr);
  Float *pmean = (Float *) mxGetData(output[0]);
  Float *pstdv = (Float *) mxGetData(output[1]);

  if(n_vec > 1) {
    size_t sz = (n_vec+1) * sizeof(Float);
    GenericHeapElement &pres = d_fast_heap->get(sz);
    Float *res = static_cast<Float*>(*pres);
 
    h_mean_stdv_vec(n_vec, p_src_vec, res);

    gpuErrChk(cudaMemcpy(pmean, res, sizeof(Float), cudaMemcpyDeviceToHost),
	      "cuda_mean_stdv_mex:memcpy", "");
    gpuErrChk(cudaMemcpy(pstdv, res+1, sizeof(Float), cudaMemcpyDeviceToHost),
	      "cuda_mean_stdv_mex:memcpy", "");

    pres.discard();
  }
  else if(n_vec == 1){
    gpuErrChk(cudaMemcpy(pmean, p_src_vec, sizeof(Float), cudaMemcpyDeviceToHost),
	      "cuda_mean_stdv_mex:memcpy", "");
    *pstdv = 0;
  }
  else {			// n_vec == 0
    *pmean = 0;
    *pstdv = 0;
  }
}

template<class Float>
static void calcCPU(const mxArray *pr,
		    size_t n_vec,
		    mxArray *output[2])
{
  const Float *p_src_vec = (const Float*) mxGetData(pr);
  Float *pmean = (Float *) mxGetData(output[0]);
  Float *pstdv = (Float *) mxGetData(output[1]);

  if(n_vec == 0) {
    *pmean = 0;
    *pstdv = 0;
  }
  else if(n_vec == 1) {
    *pmean = *p_src_vec;
    *pstdv = 0;
  }
  else {
    *pmean = c_mean_vec(n_vec, p_src_vec);
    *pstdv = c_stdv_vec(n_vec, p_src_vec, *pmean);
  }
}

static const char *errId = "cuda_mean_stdv_mex:mex_assert";

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxClassID class_id;
  size_t n_vec;
  const mxGPUArray *pr;

  TIMER_START(Timers::TIMER_MEAN_STDV);

  // Using Macro to avoid unnecessary nvcc warning (defined but not referenced)

  // Check correctness and get input information
  mex_assert((nlhs == 2 || nrhs == 1),
	     (errId,
	      "%s:%d cuda_mean_stdv_mex got %d input and %d output arguments\n"
	      "should have 1 input and 2 output arguments",
	      __FILE__, __LINE__, nrhs, nlhs));

  int is_gpu = mxIsGPUArray(prhs[0]);

  if(is_gpu) {
    pr = mxGPUCreateFromMxArray(prhs[0]);
    mex_assert((mxGPUGetComplexity(pr) == mxREAL),
	       (errId, "Input to cuda_mean_mex must be real"));
    class_id = mxGPUGetClassID(pr);

    n_vec = (size_t) mxGPUGetNumberOfElements(pr);
  }
  else {
    mex_assert(!mxIsComplex(prhs[0]), (errId, "Input must be real"));
    class_id = mxGetClassID(prhs[0]);

    n_vec = (size_t) mxGetNumberOfElements(prhs[0]);
  }

  mex_assert((class_id == mxSINGLE_CLASS || class_id == mxDOUBLE_CLASS),
	     (errId, "Input must be float (%d) of double (%d). Currnt type %d",
	      mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id));

  if(is_gpu)
    mex_assert(mxGPUisVector(pr),
	       (errId, "Input should be a vector"));
  else
    mex_assert(mxIsVector(prhs[0]),
	       (errId, "Input should be a vector"));

  plhs[0] = mxCreateNumericMatrix(1, 1, class_id, mxREAL);
  plhs[1] = mxCreateNumericMatrix(1, 1, class_id, mxREAL);
    
  if(is_gpu) { 
    if(class_id == mxSINGLE_CLASS)
      calcGPU<float>(pr, n_vec, plhs);
    else
      calcGPU<double>(pr, n_vec, plhs);
    mxGPUDestroyGPUArray(pr);
  }
  else {
    if(class_id == mxSINGLE_CLASS)
      calcCPU<float>(prhs[0], n_vec, plhs);
    else
      calcCPU<double>(prhs[0], n_vec, plhs);
  }

  TIMER_STOP(Timers::TIMER_MEAN_STDV);
}
