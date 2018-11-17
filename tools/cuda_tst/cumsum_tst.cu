/** \file
Test program for cumsum (cummulative sum)
*/
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "CudaDevInfo.h"
#include "cumsum.h"

#if 0
void h_cumsum(size_t, const double *, double *);
__global__ void d_cumsum(size_t, int, int, double *);
void h_cumsum(size_t, const float *, float *);
__global__ void d_cumsum(size_t, int, int, float *);
void h_cumsum(size_t, const long *, long *);
__global__ void d_cumsum(size_t, int, int, long *);
#endif

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
     "The program creates a random vector of size <N>, in the range [<rng0>, <rng1>].\n"
     "cumsum is computed on the vector in CPU and GPU and the results are compared.\n"
     "This is repeated for by casting the input into type double, single, and long.\n"
     "If the -v option is specified the vectors are printed out\n",
     cmd);
  exit(EXIT_FAILURE);
}

template<class T>
void run_tests(size_t n_vec,
	       const T*vec,
	       const char *type_name,
	       const char *fmt,
	       int verbose
	       ) {
  const size_t sz = n_vec * sizeof(T);
  T *h_res_gpu = (T *) malloc(sz); 
  T* d_vec;
  gpuErrChk(cudaMalloc(&d_vec, sz),"cumsum_tst:cuda_error", ""); 
  T *d_res_gpu;
  gpuErrChk(cudaMalloc(&d_res_gpu, sz), "cumsum_tst:cuda_error", ""); 
  T *res_cpu = (T *) malloc(sz);

  gpuErrChk(cudaMemcpy(d_vec, vec, sz, cudaMemcpyHostToDevice), "cumsum_tst:cuda_error", "");

  // Test cumsum
  c_cumsum(n_vec, vec, res_cpu);
  h_cumsum(n_vec, d_vec, d_res_gpu);
  gpuErrChk(cudaMemcpy(h_res_gpu, d_res_gpu, sz, cudaMemcpyDeviceToHost), "cumsum_tst:cuda_error", "");

  char fmt_str[256];
  sprintf(fmt_str, "%s cumsum test error: %s\n", type_name, fmt);
  printf(fmt_str, max_abs_vec_diff(n_vec, h_res_gpu, res_cpu));

  if(verbose) {
    size_t i,j;

    sprintf(fmt_str, "%%s%s", fmt);
    printf("Result is:");

    for(i=0; i<n_vec; i+= 5) {
      printf("\n %lu:",i);
      for(j=i; j<n_vec && j<i+5; j++)
	printf(fmt_str, (j%5)?" ":"\nvec: ", vec[j]);
      for(j=i; j<n_vec && j<i+5; j++)
	printf(fmt_str, (j%5)?" ":"\nCPU: ", res_cpu[j]);
      for(j=i; j<n_vec && j<i+5; j++)
	printf(fmt_str, (j%5)?" ":"\nGPU: ", h_res_gpu[j]);
      for(j=i; j<n_vec && j<i+5; j++)
	printf(fmt_str, (j%5)?" ":"\nDif: ", h_res_gpu[j]-res_cpu[j]);
    }
    printf("\n++++++++++++++++++\n");
  }

  free(res_cpu);
  gpuErrChk(cudaFree(d_res_gpu), "cumsum_tst:cuda_error", "");
  gpuErrChk(cudaFree(d_vec), "cumsum_tst:cuda_error", "");
  free(h_res_gpu);
}

int main(int argc, const char *argv[]) {
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
  double *dbl = (double *) malloc(n_vec*sizeof(double)); 
  float *flt = (float *) malloc(n_vec*sizeof(float)); 
  long *lng = (long *) malloc(n_vec*sizeof(long)); 

  // Initialize random data
  for(i=0; i<n_vec; i++) {
    dbl[i] = rng0 + ((rng1-rng0)/double(RAND_MAX))*double(rand());
    flt[i] = float(dbl[i]);
    lng[i] = long(dbl[i]);
  }

#if 0
  if(verbose) {
    printf("vector is:");

    for(i=0; i<n_vec; i++)
      printf("%s%12.7g", (i%5)?" ":"\n", dbl[i]);
    for(i=0; i<n_vec; i++)
      printf("%s%12.7g", (i%5)?" ":"\n", flt[i]);
    for(i=0; i<n_vec; i++)
      printf("%s%12ld", (i%5)?" ":"\n", lng[i]);
    printf("\n++++++++++++++++++\n");
  }
#endif

  run_tests(n_vec, dbl, "double", "%12.7g", verbose);
  run_tests(n_vec, flt, "float", "%12.7g", verbose);
  run_tests(n_vec, lng, "long", "%12ld", verbose);

  free(lng);
  free(flt);
  free(dbl);

  return 0;
}
