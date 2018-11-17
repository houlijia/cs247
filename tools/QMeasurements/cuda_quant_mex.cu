/**
   \file 

   MEX functions to perform quantization. 
 */

#include <string.h>
#include <math.h>

#include "mex.h"
#ifdef __NVCC__
#include "gpu/mxGPUArray.h"
#endif

#include "mex_tools.h"
#include "CudaDevInfo.h"
#include "cc_quant.h"
#include "cc_quant_mex_calc.h"
#include "mex_assert.h"
#include "fast_heap.h"
#include "mex_context.h"
#include "timers.h"

#ifdef __NVCC__
#include "cuda_quant.h"
#include "mex_gpu_tools.h"
#include "mxGetGPUVec.h"
#endif

#ifdef __NVCC
#define PROG_NAME "cuda_quant_mex"
#else
#define PROG_NAME "cc_quant_mex"
#endif

static char const * const errId = PROG_NAME ":InvalidInput";
static char const * const errIdMemcpy = PROG_NAME ":memcpy";


#ifdef __NVCC__
template<class Float, class LBL>
static void calcGPU(size_t n_no_clip, //!< No. of measurements which are not clipped.
		    size_t n_clip, //!< No. of measurements which are to be clipped.
		    const Float *vec_no_clip, //!< An array of \c n_no_clip entries to quantize
		    const Float *vec_clip, //!< An array of \c n_clip entries to quantize
		    Float intvl,
		    Float offset,
		    LBL sat_lvl,
		    int nlhs,
		    mxArray *plhs[]
		    ) {
  mxGPUArray *q_no_clip = create_GPUMatrix(n_no_clip, 1, mx_class_id<int32_T>());
  mxGPUArray *q_clip = create_GPUMatrix(n_clip, 1, mx_class_id_of(sat_lvl));
  int32_T *save = NULL;
  size_t save_cnt;
  GenericHeapElement *ppsave;
  
  if(nlhs == 3) {
    GenericHeapElement &psave = d_fast_heap->get(n_clip * sizeof(int32_T));
    save = static_cast<int32_T *>(*psave);
    ppsave = &psave;
  }

  h_quant(n_no_clip, n_clip, vec_no_clip, vec_clip, intvl, offset, sat_lvl,
	  (int32_T *)mxGPUGetData(q_no_clip), (LBL *)mxGPUGetData(q_clip),
	  save, &save_cnt);

  plhs[0] = mxGPUCreateMxArrayOnGPU(q_no_clip);
  mxGPUDestroyGPUArray(q_no_clip);
  plhs[1] = mxGPUCreateMxArrayOnGPU(q_clip);
  mxGPUDestroyGPUArray(q_clip);

  if(nlhs == 3) {
    mxGPUArray *g_save = create_GPUMatrix(save_cnt, 1, mx_class_id<int32_T>());
    gpuErrChk(cudaMemcpy(mxGPUGetData(g_save), save, save_cnt*sizeof(*save),
			 cudaMemcpyDeviceToDevice), errIdMemcpy, "");
    plhs[2] = mxGPUCreateMxArrayOnGPU(g_save);
    mxGPUDestroyGPUArray(g_save);
    ppsave->discard();
  }
}
#endif

/**
   This MEX functions performs quantization. 
   Input:
     prhs[0] - (vec_no_clip) no-clip easurements vector (single or double float)
     prhs[1] - (vec_clip) clip easurements vector (single or double float)
     prhs[2] - (intvl) quantization interval (scalar)
     prhs[3] - (offset) quantization offset (scalar)
     prhs[4] - (n_bins) number of bins (scalar)
   Output:
     plhs[0] - (q_no_clip) The quantized no-clip measurements (int32).
     plhs[1] - (q_vec) The quantized rest of measurements(int16).
     plhs[2] - (save) Saved values for clipped measurements (int32). Optional.

   Inputs are on the CPU except for prhs[0] which can be either on GPU or on
   CPU. Outputs are in the same place as the input.
 */

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  int k;
  mxClassID class_id;
  size_t n_no_clip, n_clip;
  const void *p_no_clip, *p_clip;

   TIMER_START(Timers::TIMER_QUANT);

  // Check correctness and get input information
  mex_assert(nlhs>=2 && nlhs<=3  && nrhs==5,
	     (errId, "Should have 5 input and 2 or 3 output arguments"));

  for(k=2; k<5; k++) {
#ifdef __NVCC__
    mex_assert(!mxIsGPUArray(prhs[k]),
	       (errId, "argument %d cannot be on GPU", k+1));
#endif

    mex_assert(mxIsScalar(prhs[k]),
	       (errId, "argument %d cannot be non-scalar", k+1));

    mex_assert(!mxIsComplex(prhs[k]),
	       (errId, "argument %d cannot be complex", k+1));
  }

  double intvl = mxGetScalar(prhs[2]);
  mex_assert(intvl > 0,
	     (errId, "argument 3 must be positive", k+1));
  double offset = mxGetScalar(prhs[3]);
  double val = mxGetScalar(prhs[4]);
  mex_assert(val>=1 && floor(val)==val && val < 0x10000,
    (errId, "argument 5 must be a positive 16 bit integer", k+1));
  int16_T sat_lvl = uint16_T(val+1);
  
  val = mxGetScalar(prhs[4]);
  mex_assert(val>=0 && floor(val)==val,
	     (errId, "argument 5 must be a non-negative integer", k+1));
  n_no_clip = size_t(val);
  
#ifdef __NVCC__
  int is_gpu = mxIsGPUArray(prhs[0]);

  if(is_gpu) {

    const mxGPUArray *vec_no_clip = mxGPUCreateFromMxArray(prhs[0]);
    mex_assert(mxGPUGetComplexity(vec_no_clip) == mxREAL,
	       (errId, "First argument cannot be complex"));
    n_no_clip = (size_t) mxGPUGetNumberOfElements(vec_no_clip);
    p_no_clip = mxGPUGetDataReadOnly(vec_no_clip);

    const mxGPUArray *vec_clip = mxGPUCreateFromMxArray(prhs[1]);
    mex_assert(mxGPUGetComplexity(vec_clip) == mxREAL,
	       (errId, "Second argument cannot be complex"));
    n_clip = (size_t) mxGPUGetNumberOfElements(vec_clip);
    p_clip = mxGPUGetDataReadOnly(vec_clip);

    class_id = mxGPUGetClassID(vec_no_clip);
    mex_assert(class_id == mxGPUGetClassID(vec_clip),
       (errId,
			 "Input vectors must be of the same class."
		   " No clip: %d clip: %d", class_id, mxGPUGetClassID(vec_clip)));

    mex_assert(class_id == mxDOUBLE_CLASS || class_id == mxSINGLE_CLASS,
	       (errId,
			"first argument  must be float (%d) of double (%d). Currnt type %d",
			   mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id));
    switch(class_id) {
    case mxDOUBLE_CLASS:
      calcGPU(n_no_clip, n_clip, (double *)p_no_clip, (double *)p_clip, 
	      intvl, offset, sat_lvl, nlhs, plhs);
      break;
    case mxSINGLE_CLASS:
      calcGPU(n_no_clip, n_clip, (float *)p_no_clip, (float *)p_clip,
	      (float)intvl, (float)offset, sat_lvl, nlhs, plhs);
      break;
     }
    mxGPUDestroyGPUArray(vec_clip);
    mxGPUDestroyGPUArray(vec_no_clip);
  }
  else {
#endif  /* __NVCC__ */
    mex_assert(mxGetClassID(prhs[0]) == mxGetClassID(prhs[1]),
	       (errId, "Input vectors must be of the same class"));
    mex_assert(!mxIsComplex(prhs[0]) && !mxIsComplex(prhs[1]),
	       (errId, "Input vectors cannot be complex"));
    n_no_clip = (size_t) mxGetNumberOfElements(prhs[0]);
    n_clip = (size_t) mxGetNumberOfElements(prhs[1]);
    class_id = mxGetClassID(prhs[0]);

    p_no_clip = mxGetData(prhs[0]);
    p_clip = mxGetData(prhs[1]);

    mex_assert(class_id == mxDOUBLE_CLASS || class_id == mxSINGLE_CLASS,
	       (errId,
		"first argument  must be float (%d) of double (%d). Currnt type %d",
		mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id));
    switch(class_id) {
    case mxDOUBLE_CLASS:
      calcCPU(n_no_clip, n_clip, (double *)p_no_clip, (double *)p_clip,
	      intvl, offset, sat_lvl, nlhs, plhs);
      break;
    case mxSINGLE_CLASS:
      calcCPU(n_no_clip, n_clip, (float *)p_no_clip, (float *)p_clip,
	      (float)intvl, (float)offset, sat_lvl, nlhs, plhs); 
      break;
   }
#ifdef __NVCC__
  }
#endif

  TIMER_STOP(Timers::TIMER_QUANT);

}
