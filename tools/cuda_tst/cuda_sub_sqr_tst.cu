#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "CudaDevInfo.h"
#include "cuda_sub_sqr.h"
#include "cc_sub_sqr.h"
#include "fast_heap.h"

template <class T>
T max_abs_vec_diff(size_t n_vec,	//!< no of elements
		   const T *vec1,	//!< The vector
		   const T *vec2	//!< The vector
	      ) {
  T mx = fabs(double(vec1[0]) - double(vec2[0]));
  size_t i;
  
  for(i=1; i<n_vec; i++) {
    T df = fabs(double(vec1[i]) - double(vec2[i]));
    if(df > mx)
      mx = df;
  }

  return mx;
};

int main(int argc, const char *argv[]) {
  int err = EXIT_SUCCESS;
  int nv;
  size_t n_vec;
  const double dbl_sb = double(rand())/double(RAND_MAX); 
  const float flt_sb = float(dbl_sb);
  size_t size_d, size_f;
  size_t i;

  if(argc!=2 || sscanf(argv[1],"%d", &nv) != 1 || nv<= 0) {
    fprintf(stderr, "USAGE:\n"
	   "    %s <N>   (<N> > 0)\n"
	   "The functions creates a random number sbval and random vector vec of lenght N.\n"
	   "For each element vec[i] it computes (vec[i]-sbval)^2 with and without the GPU\n"
	   "and reports the error. This is repeated for double and float computation\n",
	   argv[0]);
    exit(EXIT_FAILURE);
  }

  n_vec = size_t(nv);

  switch(err) {
  case cudaSuccess: 
    break;
  case cudaErrorInvalidDevice:
    fprintf(stderr,"h_cuda_init error: INvalid Device\n");
    exit(err);
  default:
    fprintf(stderr, "Unexpected error in h_cuda_init: %d\n", err);
    exit(err);
  }
  
  // Memory allocation
  size_d = n_vec * sizeof(double);

  GenericHeapElement &pdbl = fast_heap->get(size_d);
  double *dbl = static_cast<double *>(*pdbl);

  GenericHeapElement &ph_dbl_cpu = h_fast_heap->get(size_d);
  double *h_dbl_gpu = static_cast<double *>(*ph_dbl_cpu);

  GenericHeapElement &pd_dbl_gpu = d_fast_heap->get(size_d);
  double *d_dbl_gpu = static_cast<double *>(*pd_dbl_gpu);

  GenericHeapElement &pd_dbl_sb = d_fast_heap->get(sizeof(double));
  double *d_dbl_sb = static_cast<double *>(*pd_dbl_sb);

  GenericHeapElement &pdbl_cpu = fast_heap->get(size_d);
  double *dbl_cpu = static_cast<double *>(*pdbl_cpu);

  size_f = n_vec * sizeof(float);

  GenericHeapElement &pflt = fast_heap->get(size_f);
  float *flt = static_cast<float *>(*pflt);

  GenericHeapElement &ph_flt_cpu = h_fast_heap->get(size_f);
  float *h_flt_gpu = static_cast<float *>(*ph_flt_cpu);

  GenericHeapElement &pd_flt_gpu = d_fast_heap->get(size_f);
  float *d_flt_gpu = static_cast<float *>(*pd_flt_gpu);

  GenericHeapElement &pd_flt_sb = d_fast_heap->get(sizeof(float));
  float *d_flt_sb = static_cast<float *>(*pd_flt_sb);

  GenericHeapElement &pflt_cpu = fast_heap->get(size_f);
  float *flt_cpu = static_cast<float *>(*pflt_cpu);

  flt = (float *) malloc(size_f); 
  h_flt_gpu = (float *) malloc(size_f); 
  gpuErrChk(cudaMalloc(&d_flt_gpu, size_f),"cuda_sub_sqr_tst:CudaError", ""); 
  gpuErrChk(cudaMalloc(&d_flt_sb, sizeof(float)),"cuda_sub_sqr_tst:CudaError", "");
  flt_cpu = (float *) malloc(size_f);

  // Initialize random data
  for(i=0; i<n_vec; i++) {
    dbl[i] = double(rand())/double(RAND_MAX); 
    flt[i] = float(dbl[i]);
  }
			     
  // Double test
  gpuErrChk(cudaMemcpy(d_dbl_gpu, dbl, size_d, cudaMemcpyHostToDevice),"cuda_sub_sqr_tst:CudaError", "");
  gpuErrChk(cudaMemcpy(d_dbl_sb, &dbl_sb, sizeof(dbl_sb), cudaMemcpyHostToDevice),"cuda_sub_sqr_tst:CudaError", "");
  h_sub_sqr<double>(d_dbl_sb, d_dbl_gpu, n_vec, d_dbl_gpu);
  gpuErrChk(cudaMemcpy(h_dbl_gpu, d_dbl_gpu, size_d, cudaMemcpyDeviceToHost),"cuda_sub_sqr_tst:CudaError", "");
  c_sub_sqr(dbl_sb, dbl, n_vec, dbl_cpu);
  printf("Max. Abs. difference in double computation using reference: %8.3g\n", 
	 max_abs_vec_diff<double>(n_vec, dbl_cpu, h_dbl_gpu));

  gpuErrChk(cudaMemcpy(d_dbl_gpu, dbl, size_d, cudaMemcpyHostToDevice),"cuda_sub_sqr_tst:CudaError", "");
  h_sub_sqr<double>(dbl_sb, d_dbl_gpu, n_vec, d_dbl_gpu);
  gpuErrChk(cudaMemcpy(h_dbl_gpu, d_dbl_gpu, size_d, cudaMemcpyDeviceToHost),"cuda_sub_sqr_tst:CudaError", "");

  c_sub_sqr(dbl_sb, dbl, n_vec, dbl_cpu);
  printf("Max. Abs. difference in double computation: using direct value %8.3g\n", 
	 max_abs_vec_diff<double>(n_vec, dbl_cpu, h_dbl_gpu));
  
  // float test
  gpuErrChk(cudaMemcpy(d_flt_gpu, flt, size_f, cudaMemcpyHostToDevice),"cuda_sub_sqr_tst:CudaError", "");
  gpuErrChk(cudaMemcpy(d_flt_sb, &flt_sb, sizeof(flt_sb), cudaMemcpyHostToDevice),"cuda_sub_sqr_tst:CudaError", "");
  h_sub_sqr<float>(d_flt_sb, d_flt_gpu, n_vec, d_flt_gpu);
  gpuErrChk(cudaMemcpy(h_flt_gpu, d_flt_gpu, size_f, cudaMemcpyDeviceToHost),"cuda_sub_sqr_tst:CudaError", "");

  c_sub_sqr<float>(flt_sb, flt, n_vec, flt_cpu);
  printf("Max Abs. difference in float computation using reference: %8.3g\n", 
	 max_abs_vec_diff<float>(n_vec, flt_cpu, h_flt_gpu));
  
  gpuErrChk(cudaMemcpy(d_flt_gpu, flt, size_f, cudaMemcpyHostToDevice),"cuda_sub_sqr_tst:CudaError", "");
  h_sub_sqr<float>(flt_sb, d_flt_gpu, n_vec, d_flt_gpu);
  gpuErrChk(cudaMemcpy(h_flt_gpu, d_flt_gpu, size_f, cudaMemcpyDeviceToHost),"cuda_sub_sqr_tst:CudaError", "");

  c_sub_sqr<float>(flt_sb, flt, n_vec, flt_cpu);
  printf("Max Abs. difference in float computation using direct value: %8.3g\n", 
	 max_abs_vec_diff<float>(n_vec, flt_cpu, h_flt_gpu));
  
  // Free memory
  pdbl.discard(); ph_dbl_cpu.discard(); pd_dbl_gpu.discard(); pd_dbl_sb.discard(); pdbl_cpu.discard();
  pflt.discard(); ph_flt_cpu.discard(); pd_flt_gpu.discard(); pd_flt_sb.discard(); pflt_cpu.discard();
 
  return EXIT_SUCCESS;
}
