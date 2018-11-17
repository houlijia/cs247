#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "CudaDevInfo.h"
#include "cuda_vec_op_unary.h"
#include "cuda_vec_op_scalar.h"
#include "cuda_sum_mean_var.h"
#include "cc_vec_op_unary.h"
#include "cc_vec_op_scalar.h"
#include "cc_sum_mean_var.h"
#include "fast_heap.h"

#include "cuda_vec_do.h"

template <class T>
T max_abs_vec_diff(size_t n_vec,	//!< no of elements
		   const T *vec1,	//!< The vector
		   const T *vec2	//!< The vector
		   ) {
  T mx = fabs(double(vec1[0]) - double(vec2[0]));
  size_t i;
  
  for(i=1; i<n_vec; i++) {
    T df = fabs(vec1[i] - vec2[i]);
    if(df > mx)
      mx = df;
  }

  return mx;
}

static void help(const char *cmd)
{
  fprintf
    (stderr, "USAGE:\n"
     "    %s [-v] <N>   <rng0> <rng1>\n"
     "where <N> is a positive integer and <rng0>, <rng1> are floating point numbers.\n" 
     "The program creates a scalar and vector of size <N>, both random in the range\n"
     "between <rng0> and <rng1>. Then various tests are run and the error is reported.\n"
     "This is repeated for double and float computation\n",
     cmd);
  exit(EXIT_FAILURE);
}

template<class T>
void run_tests(size_t n_vec, T sclr, const T*vec, const char *type_name) {
  const size_t sz = n_vec * sizeof(T);
  const size_t sz1 = (n_vec+1) * sizeof(T);
  
  GenericHeapElement &ph_res_gpu = h_fast_heap->get(sz);
  T *h_res_gpu = static_cast<T *>(*ph_res_gpu);

  GenericHeapElement &pd_vec = d_fast_heap->get(sz);
  T *d_vec = static_cast<T *>(*pd_vec);

  GenericHeapElement &pd_res_gpu = d_fast_heap->get(sz1);
  T *d_res_gpu = static_cast<T *>(*ph_res_gpu);

  GenericHeapElement &pres_cpu = fast_heap->get(sz1);
  T *res_cpu = static_cast<T *>(*pres_cpu);

  gpuErrChk(cudaMemcpy(d_vec, vec, sz, cudaMemcpyHostToDevice),"cuda_tst_do:cuda_error", "");

  // Test sum
  res_cpu[0] = c_sum_vec(n_vec, vec);
  h_sum_vec(n_vec, d_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sizeof(T), cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

  printf("%s sum test: %g\n", type_name, max_abs_vec_diff(1, h_res_gpu, res_cpu));

  // Test Mean and standard deviation
  res_cpu[0] = c_mean_vec(n_vec, vec);
  res_cpu[1] = c_stdv_vec(n_vec, vec, res_cpu[0]);
  h_mean_vec(n_vec, d_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sizeof(T), cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

  printf("%s mean test: %g\n", type_name, max_abs_vec_diff(1, h_res_gpu, res_cpu));

  if(n_vec > 1) {
    h_stdv_vec(n_vec, d_vec, d_res_gpu, d_res_gpu+1);
    gpuErrChk(cudaMemcpy(h_res_gpu+1, d_res_gpu+1, sizeof(T), cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

    printf("%s stdv test: %g\n", type_name, max_abs_vec_diff(1, h_res_gpu+1, res_cpu+1));

    h_mean_stdv_vec(n_vec, d_vec, d_res_gpu);
    gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, 2*sizeof(T), cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

    printf("%s combined mean test: %g\n", type_name, max_abs_vec_diff(1, h_res_gpu, res_cpu));
    printf("%s combined stdv test: %g\n", type_name, max_abs_vec_diff(1, h_res_gpu+1, res_cpu+1));
  }

  // Test sqrt and abs
  c_vec_abs(vec, n_vec, res_cpu);
  c_vec_sqrt(res_cpu, n_vec, res_cpu);
  h_vec_abs(d_vec, n_vec, d_res_gpu);
  h_vec_sqrt(d_res_gpu, n_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sz, cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

  printf("%s sqrt(abs()) test: %g\n", type_name, max_abs_vec_diff(n_vec, h_res_gpu, res_cpu));
  
  // Test subtract scalar
  c_vec_sub_scalar(sclr, vec, n_vec, res_cpu);
  h_vec_sub_scalar(sclr, d_vec, n_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sz, cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

  printf("%s sub_scalar test: %g\n", type_name, max_abs_vec_diff(n_vec, h_res_gpu, res_cpu));
  
  // Test add scalar
  c_vec_add_scalar(sclr, vec, n_vec, res_cpu);
  h_vec_add_scalar(sclr, d_vec, n_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sz, cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

  printf("%s add_scalar test: %g\n", type_name, max_abs_vec_diff(n_vec, h_res_gpu, res_cpu));
  
  // Test multiply by scalar
  c_vec_mlt_scalar(sclr, vec, n_vec, res_cpu);
  h_vec_mlt_scalar(sclr, d_vec, n_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sz, cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

  printf("%s mlt_scalar test: %g\n", type_name, max_abs_vec_diff(n_vec, h_res_gpu, res_cpu));
  
  // Test divide by scalar
  if(sclr != 0) {
    c_vec_div_scalar(sclr, vec, n_vec, res_cpu);
    h_vec_div_scalar(sclr, d_vec, n_vec, d_res_gpu);
    gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sz, cudaMemcpyDeviceToHost),"cuda_tst_do:cuda_error", "");

    printf("%s div_scalar test: %g\n", type_name, max_abs_vec_diff(n_vec, h_res_gpu, res_cpu));
  }

  pres_cpu.discard();
  pd_res_gpu.discard();
  pd_vec.discard();
  ph_res_gpu.discard();
}

int cuda_vec_do(int argc, const char *argv[]) {
  int nv;
  double rng0, rng1;
  size_t i;
  int verbose;

  if(argc < 4)
    help(argv[0]);

  verbose = !strcmp(argv[1],"-v");

  if(argc-verbose != 4 ||
     (sscanf(argv[verbose+1],"%d",&nv) +
      sscanf(argv[verbose+2],"%lg",&rng0) + sscanf(argv[verbose+3],"%lg",&rng1) != 3) ||
     nv <= 0
     )
    help(argv[0]);

  // Memory allocation and initialize random data
  const size_t n_vec = nv;
  const double dbl_sclr = rng0 + ((rng1-rng0)/double(RAND_MAX))*double(rand()); 
  const float flt_sclr = float(dbl_sclr);
  double *dbl = (double *) malloc(n_vec*sizeof(double)); 
  float *flt = (float *) malloc(n_vec*sizeof(float)); 

  // Initialize random data
  for(i=0; i<n_vec; i++) {
    dbl[i] = rng0 + ((rng1-rng0)/double(RAND_MAX))*double(rand());
    flt[i] = float(dbl[i]);
  }

  if(verbose) {
    printf("scalar is: %g\n"
	   "vector is:", dbl_sclr);

    for(i=0; i<n_vec; i++)
      printf("%s%g", (i%8)?" ":"\n", dbl[i]);
    printf("\n++++++++++++++++++\n");
  }

  run_tests(n_vec, dbl_sclr, dbl, "double");
  run_tests(n_vec, flt_sclr, flt, "float");

  free(flt);
  free(dbl);

  return 0;
}
