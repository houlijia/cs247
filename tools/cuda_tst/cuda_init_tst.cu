#include <stdio.h>
#include <stdlib.h>

#
#include "CudaDevInfo.h"

int main(int argc, const char *argv[]) {
  int err;
  int dev_num = 1;

  if(argc!=2 || sscanf(argv[1],"%d", &dev_num) != 1) {
    fprintf(stderr, "USAGE:\n"
	   "    %s <device number>\n",
	   argv[0]);
    exit(EXIT_FAILURE);
  }

  err = cudaSetDevice(dev_num);
  switch(err) {
  case cudaSuccess: 
    printf("cudaSetDevice(%d) successful\n", dev_num);
    break;
  case cudaErrorInvalidDevice:
    fprintf(stderr, "Invalid device: %d\n", dev_num);
    exit(err);
  case cudaErrorDeviceAlreadyInUse:
    fprintf(stderr, "Invalid device: %d\n", dev_num);
    exit(err);
  default:
    fprintf(stderr, "Unexpected error in cudaSetDevice(%d): %d\n", dev_num, err);
    exit(err);
  }

  printf("Name: %s\n"
	 "Global Memory: %lu B\n"
	 "Max threads per block: %d\n"
	 "Max threads dim: [%d %d %d]\n"
	 "Max grid size: [%d %d %d]\n",
	 cuda_dev->getProp().name,
	 (unsigned long) cuda_dev->getProp().totalGlobalMem,
	 cuda_dev->getProp().maxThreadsPerBlock,
	 cuda_dev->getProp().maxThreadsDim[0], cuda_dev->getProp().maxThreadsDim[1], 
	 cuda_dev->getProp().maxThreadsDim[2], 
	 cuda_dev->getProp().maxGridSize[0], cuda_dev->getProp().maxGridSize[1], 
	 cuda_dev->getProp().maxGridSize[2]);

  return err;
}
