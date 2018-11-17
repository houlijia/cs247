#define REPORT_CUDA_INIT 0	/**< If true, report when initializing */
#define RESET_CUDA_INIT 0	/**< If true, reset device when initializing */

#if REPORT_CUDA_INIT
#include <stdio.h>
#endif

#ifdef MATLAB_MEX_FILES
#include "mex.h"
#define printf mexPrintf
#endif

#include "CudaDevInfo.h"

cudaError_t h_cuda_init(int dev_no /**< negative = find out the device */
			) {
  cudaError err;

  cuda_dev = &cuda_dev_dt;

  if (cuda_dev != NULL && cuda_dev_dt.dev_indx >= 0)
    return cudaSuccess;

  if(dev_no>= 0)
    cuda_dev_dt.dev_indx = dev_no;
  else {
    err = cudaGetDevice(&cuda_dev_dt.dev_indx);
    if (err != cudaSuccess)
      return err;
  }

#if RESET_CUDA_INIT
  gpuErrChk(cudaDeviceReset(),"","");
#endif

#if REPORT_CUDA_INIT
  printf("Initializing CUDA cuda_dev=0x%lX\n", (unsigned long)cuda_dev);
  printf("Device Index: %d\n", cuda_dev->dev_indx);
#endif
  err = cudaGetDeviceProperties(&cuda_dev_dt.prop, cuda_dev_dt.dev_indx);

  cuda_dev_dt.log2_thrds_blk = 0;
  for(int m=cuda_dev_dt.prop.maxThreadsPerBlock; m>1; m >>= 1)
    cuda_dev_dt.log2_thrds_blk++;

  // Force some action
  void *p;
  const size_t len = 10;
  gpuErrChk(cudaMalloc(&p , len), "Alloc error", "");
  gpuErrChk(cudaMemset(p, 0, len), "memset error", "");
  gpuErrChk(cudaFree(p), "Alloc error", "");

  return err;		      
}


