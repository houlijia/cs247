#ifndef __MEX_GPU_TOOLS_HDR__
#define __MEX_GPU_TOOLS_HDR__

#include "mex.h"
#include "gpu/mxGPUArray.h"

mxGPUArray *create_GPUMatrix(mwSize n_rows,
			     mwSize n_cols,
			     mxClassID cid,
			     mxComplexity cmplx = mxREAL
			     );

//* Returns the dim-th dimension of a GPU array (0 if out of range)
//* (not efficient, but supposedly rarely used)
inline mwSize mxGPUGetDim(const mxGPUArray *pg,
			int dim) //*< Index of the dimension
{
  if(mxGPUGetNumberOfDimensions(pg) < dim)
    return 0;
  const mwSize *pdim = mxGPUGetDimensions(pg); 
  mwSize result = pdim[dim];
  mxFree((void *)pdim);
  return result;
}

//* Returns the number of rows and columns of a 2D GPU array 
inline void mxGPUGetMatDims(const mxGPUArray *pg, //*< input GPU array
			    mwSize &nr,		  //*< returns number of rows
			    mwSize &nc		  //*< returns number of columns
			    )
{
  const mwSize *pdim = mxGPUGetDimensions(pg);
  nr = pdim[0];
  nc = pdim[1];
  mxFree((void *)pdim);
}

// Return true if the GPU array \c *pg is a vector
inline bool mxGPUisVector(const mxGPUArray *pg)
{
  if(mxGPUGetNumberOfDimensions(pg) != 2)
    return false;
  const mwSize *pdim = mxGPUGetDimensions(pg); 
  bool result = (pdim[0] == 1 || pdim[1] == 1);
  mxFree((void *)pdim);
  return result;
}
  
//* Return true the GPU array \c *pg first two dimensions equal D0 and D1
inline bool mxGPUEqualMatDims(const mxGPUArray *pg,
			 mwSize D0,	      
			 mwSize D1
			 )
{
  const mwSize *pdim = mxGPUGetDimensions(pg); 
  bool result = (pdim[0] == D0 && pdim[1] == D1);
  mxFree((void *)pdim);
  return result;
}
 
//* Return true the GPU array \c *pg has two dimension and they equal D0 and D1
inline bool mxGPU2DEqual(const mxGPUArray *pg,
			 mwSize D0,	      
			 mwSize D1
			 )
{
  return (mxGPUGetNumberOfDimensions(pg) == 2) && mxGPUEqualMatDims(pg, D0, D1);
}
 
#endif	/*  __MEX_GPU_TOOLS_HDR__ */
