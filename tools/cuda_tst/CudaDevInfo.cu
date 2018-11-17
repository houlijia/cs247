/** 
    \file
*/

#include "CudaDevInfo.h"

#define REPORT_CUDA_INIT 0	/**< If true, report when initializing */
#define RESET_CUDA_INIT 0	/**< If true, reset device when initializing */

int GPUInfo::getDeviceIndex()
{
  int indx;
 
  gpuErrChk(cudaGetDevice(&indx), "GPUInfo:getDeviceIndex:error", "Unexpected error");

  return indx;
}

int GPUInfo::deviceCount()
{
  int ndev;
 
  gpuErrChk(cudaGetDeviceCount(&ndev), "GPUInfo:deviceCount:error", "Unexpected error");

  return ndev;
}

void GPUInfo::init(int dev_ind)
{
  gpuErrChk(cudaSetDevice(dev_ind),
	     "GPUInfo:init:cudaSetDevice","Error setting GPU device");
  dev_indx = getDeviceIndex();
	     
#if RESET_CUDA_INIT
  gpuErrChk(cudaDeviceReset(),"","");
#endif

#if REPORT_CUDA_INIT
  printf("Initializing CUDA cuda_dev=0x%lX\n", (unsigned long)cuda_dev);
  printf("Device Index: %d\n", cuda_dev->dev_indx);
#endif

  gpuErrChk(cudaGetDeviceProperties(&prop, dev_indx),
	     "GPUInfo:init:cudaGetDeviceProperties","Error getting GPU properties");

  // Compute log2_thrds_blk
  log2_thrds_blk = 0;
  for(int m=prop.maxThreadsPerBlock; m>1; m >>= 1)
    log2_thrds_blk++;

  // Force some action
  void *p;
  const size_t len = 10;
  gpuErrChk(cudaMalloc(&p , len), "", "Alloc error");
  gpuErrChk(cudaMemset(p, 0, len), "", "memset error");
  gpuErrChk(cudaFree(p), "", "Alloc error");

#ifdef MATLAB_MEX_FILES

  gpuErrChk((mxInitGPU() == MX_GPU_SUCCESS)? cudaSuccess: cudaErrorInitializationError,
	     ("GPUInfo:init:mxInitGPU", "mxInitGPU failed"));

#endif
}

GlobalPtr<GPUInfo> cuda_dev("cuda_dev");

