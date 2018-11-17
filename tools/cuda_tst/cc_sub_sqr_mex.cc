/**
   This function takes a scalar prhs[0] and a vector prhs[1] and returns in
   plhs[0] the vector resulting from subtracting the scalar from the vector
   and then taking square of each element. Computation is done on the CPU, and
   result is returned in the CPU, if the inputs are on the GPU. Comptutation
   is in single or double precision float, according to the precision of the input.
 */
#include "mex.h"
#include "cc_sub_sqr.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  char const * const errId = "cuda_sub_sqr_mex:InvalidInput";
  int k;
  mxClassID class_id[2];
  size_t n_vec, n0;
  const void *p_sclr, *p_src_vec;
  void *p_dst_vec;
  
  // Check correctness and get input information
  if(nlhs != 1 || nrhs != 2)
    mexErrMsgIdAndTxt(errId,
		      "cuda_sub_sqr_mex should have 2 inputs and 1 output arguments");

  for(k=0; k<2; k++) {
      if(mxIsComplex(prhs[k]))
          mexErrMsgIdAndTxt(errId, "Inputs to cuda_sub_sqr_mex must be real");
      class_id[k] = mxGetClassID(prhs[k]);
  }
  n0 = (size_t) mxGetNumberOfElements(prhs[0]);
  
  if(class_id[0] != class_id[1])
    mexErrMsgIdAndTxt(errId, "Inputs must be of the same type");

  if(class_id[0] != mxDOUBLE_CLASS && class_id[0] != mxSINGLE_CLASS)
    mexErrMsgIdAndTxt(errId, "Inputs must be float (%d) or double (%d). Currnt type %d",
		      mxSINGLE_CLASS, mxDOUBLE_CLASS, class_id[0]);

  if(n0 != 1)
    mexErrMsgIdAndTxt(errId, "First input arguments must be a scalar");

  // Do the actual processing
  p_sclr = mxGetData(prhs[0]);
  p_src_vec = mxGetData(prhs[1]);
  n_vec = (size_t) mxGetNumberOfElements(prhs[1]);
  plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[1]), 
                                 mxGetDimensions(prhs[1]),
                                 class_id[0], 
                                 mxREAL);
  p_dst_vec = mxGetData(plhs[0]);

  switch(class_id[0]) {
  case mxDOUBLE_CLASS:
      c_sub_sqr<double>(*(double *)p_sclr, (const double *)p_src_vec, n_vec,
                        (double *)p_dst_vec);
  	  break;
  case mxSINGLE_CLASS:
      c_sub_sqr<float>(*(float *)p_sclr, (const float *)p_src_vec, n_vec,
                       (float *)p_dst_vec);
      break;
  }
}

