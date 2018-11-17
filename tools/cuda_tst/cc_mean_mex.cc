/**
   This function computes the mean of a vector given in prhs[0] and returns
   the result in plhs[0]. Comptutation is in single or double precision float,
   according to the precision of thye input.  

   If the input is empty the output is 0.
*/
#include "mex.h"
#include "cc_sum_mean_var.h"

static char const * const errId = "cc_mean_mex:InvalidInput";

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
  

  // Check correctness and get input information
  if(nlhs != 1 || nrhs != 1)
      mexErrMsgIdAndTxt(errId,
                        "cc_mean_mex should have 1 input and 1 output arguments");

  if(mxIsComplex(prhs[0]))
      mexErrMsgIdAndTxt(errId, "Input to cuda_mean_mex must be real");
  class_id = mxGetClassID(prhs[0]);

  n_vec = (size_t) mxGetNumberOfElements(prhs[0]);
  n_dim = (size_t) mxGetNumberOfDimensions(prhs[0]);
  n_rows = mxGetM(prhs[0]);
  n_cols = mxGetN(prhs[0]);

  if(n_dim != 2 ||  (n_rows > 1 && n_cols > 1))
      mexErrMsgIdAndTxt(errId,
                        "Input should be a non-empty vector");
    
  if(class_id == mxSINGLE_CLASS || class_id == mxDOUBLE_CLASS)
    plhs[0] = mxCreateNumericMatrix(1, 1, class_id, mxREAL);
  else
    mexErrMsgIdAndTxt(errId,
		      "Input must be float (%d) of double (%d). Currnt type %d",
		      mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id);

  if(class_id == mxSINGLE_CLASS)
    calcCPU<float>(prhs[0], n_vec, plhs[0]);
  else
    calcCPU<double>(prhs[0], n_vec, plhs[0]);
}
