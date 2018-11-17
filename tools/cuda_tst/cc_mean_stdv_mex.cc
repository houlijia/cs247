/**
   This MEX function computes the mean and standard deviation of a vector
   given in prhs[0] and returns the results in CPU. prhs[0] is the mean and
   prhs[1] is the standard variation. The input can be double or single
   precision. If the input is double or single the computation is in double or
   single precision, and the results are double or single, respectively.

   If the input is empty both outputs are 0. If it has one entry, only the mean is
   computed and the standard deviation is set to zero.
   computed and the standard deviation is set to zero.
*/
#include "mex.h"
#include "cc_sum_mean_var.h"
#include "mex_assert.h"
#include "timers.h"

static char const * const errId = "cc_mean_stdv_mex:InvalidInput";

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

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxClassID class_id;
  size_t n_vec;
  
  TIMER_START(Timers::TIMER_MEAN_STDV);

  // Check correctness and get input information
  mex_assert((nlhs == 2 || nrhs == 1),
	     (errId,"cuda_mean_stdv_mex should have 1 input and 2 output arguments"));
  mex_assert(!mxIsComplex(prhs[0]),
	     (errId, "Input must be real"));
  class_id = mxGetClassID(prhs[0]);

  n_vec = (size_t) mxGetNumberOfElements(prhs[0]);

  mex_assert((mxGetNumberOfDimensions(prhs[0]) == 2 &&
	      (mxGetM(prhs[0]) == 1 || mxGetN(prhs[0]) ==1)),
	     (errId,"Input should be a vector"));

  mex_assert((class_id == mxSINGLE_CLASS || class_id == mxDOUBLE_CLASS),
	     (errId, "Input must be float (%d) of double (%d). Currnt type %d",
	      mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id));

  plhs[0] = mxCreateNumericMatrix(1, 1, class_id, mxREAL);
  plhs[1] = mxCreateNumericMatrix(1, 1, class_id, mxREAL);
  
  if(class_id == mxSINGLE_CLASS)
    calcCPU<float>(prhs[0], n_vec, plhs);
  else
    calcCPU<double>(prhs[0], n_vec, plhs);

  TIMER_STOP(Timers::TIMER_MEAN_STDV);
}
