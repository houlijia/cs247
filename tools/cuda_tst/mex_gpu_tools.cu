#include "mex.h"
#include "gpu/mxGPUArray.h"

mxGPUArray *create_GPUMatrix(mwSize n_rows,
			     mwSize n_cols,
			     mxClassID cid,
			     mxComplexity cmplx
			     )
{
  const mwSize dims[2] = {n_rows, n_cols};
  mxGPUArray *B = mxGPUCreateGPUArray(2, dims, cid, cmplx, MX_GPU_DO_NOT_INITIALIZE);

  return B;
}  

