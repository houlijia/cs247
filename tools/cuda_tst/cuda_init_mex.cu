/**
   \file

   Initializing GPU functions. Returns a pointer the GPUInfo of the currnt device.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "CudaDevInfo.h"


//* Implementation of a Matlab function which receives one argument - GPU
//* index, set the device to that index (if possible) and returns the actual
//* GPU index. Note that the received and returned GPU indices begin at 1.
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  char const * const errId = "cuda_init_mex:InvalidInput";

  mex_assert(nlhs != 0 || nrhs != 1,
	     (errId, "cuda_init_mex should have one input and one output arguments"));

  if(mxInitGPU() != MX_GPU_SUCCESS)
    mexErrMsgIdAndTxt(errId, "mxInitGPU() failed");

  int indx = (int(mxGetScalar(prhs[0]))-1);

  mex_assert(indx >= 0 && indx < cuda_dev->deviceCount(),
	     ("cuda_init_mex:Illegal_arg", "Illegal Matlab GPU index: %d\n", indx+1));

  cuda_dev->setDeviceIndex(indx);

#if 0
  mexPrintf("Device index: %d\n", (int)cuda_dev->deviceIndex());
  mexPrintf("Name: %s\n"
	    "Global Memory: %lu B\n"
	    "Max threads per block: %d\n"
	    "Max threads dim: [%d %d %d]\n"
	    "Max grid size: [%d %d %d]\n",
	    cuda_dev->prop.name,
	    (unsigned long) cuda_dev->prop.totalGlobalMem,
	    cuda_dev->prop.maxThreadsPerBlock,
	    cuda_dev->prop.maxThreadsDim[0], cuda_dev->prop.maxThreadsDim[1], 
	    cuda_dev->prop.maxThreadsDim[2], 
	    cuda_dev->prop.maxGridSize[0], cuda_dev->prop.maxGridSize[1], 
	    cuda_dev->prop.maxGridSize[2]);
#endif

  plhs[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
  *mxGetPr(plhs[0]) = double(cuda_dev->deviceIndex() + 1);
}
