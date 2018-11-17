#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "CudaDevInfo.h"
#include "cuda_real_dft_sort.h"
#include "cc_real_dft_sort.h"

static int verbose;

static void help(const char *cmd)
{
  fprintf
    (stderr, "USAGE:\n"
     "    %s [-v] <N> <M>\n"
     "<N> is a positive integer specifying the DFT order.\n" 
     "<M> is the number of columns on which DFT was done (positive)\n"
     "The program creates a Floating point random array of length N, unsorts it and\n"
     "sorts it back and checks that the original sequence was received.\n"
     "If -v is specified the sequences are printed out.\n"
     "The tests are repeated for both for double and for float computation\n",
     cmd);
  exit(EXIT_FAILURE);
}

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
}

template <class Float>
void run_tests(size_t N, size_t M, const Float *vec, const char *type)
{
  size_t k, j, c;

  printf("********** %s tests **************\n", type);

  if(verbose) {
    printf("Input:");
    for(c=0; c<M; c++) {
      printf("\n Column %lu:", (unsigned long)c);
      for(k=0; k<N; k++)
	printf("%s%7.4f", (k%10)?" ":"\n", vec[c*N+k]);
    }
    printf("\n");
  }

  // Unsort tests

  Float *rl = (Float *) malloc(N*M*sizeof(Float));
  Float *im = (Float *) malloc(N*M*sizeof(Float));
  Float *d_vec;
  gpuErrChk(cudaMalloc(&d_vec, N*M*sizeof(Float)), 
	    "cuda_real_dft_sort_tst:cuda_error", "");
  gpuErrChk(cudaMemcpy(d_vec, vec, N*M*sizeof(Float), cudaMemcpyHostToDevice),
	    "cuda_real_dft_sort_tst:cuda_error", "");

  c_real_dft_unsort(N, M, vec, rl, im);

  if(verbose) {
    for(c=0; c<M; c++) {
      printf("\n CPU Column %lu:", (unsigned long)c);
      for(k=0; k<N; k+=10) {
	for(j=k; j<(N<(k+10)?N:k+10); j++)
	  printf("%s%7.4f", (j%10)?" ":"\nReal: ", rl[c*N+j]);
	for(j=k; j<(N<(k+10)?N:k+10); j++)
	  printf("%s%7.4f", (j%10)?" ":"\nImag: ", im[c*N+j]);
      }
    }
    printf("\n");
  }

  Float *d_cf;
  gpuErrChk(cudaMalloc(&d_cf, 2*N*M*sizeof(Float)),
	    "cuda_real_dft_sort_tst:cuda_error", "");
  Float *h_cf = (Float *) malloc(2*N*M*sizeof(Float));
  Float *h_rl = (Float *) malloc(N*M*sizeof(Float));
  Float *h_im = (Float *) malloc(N*M*sizeof(Float));

  h_real_dft_unsort(N, M, d_vec, d_cf);
  gpuErrChk(cudaMemcpy(h_cf, d_cf, 2*N*M*sizeof(Float), cudaMemcpyDeviceToHost),
	    "cuda_real_dft_sort_tst:cuda_error", "");
  for(k=0; k<N*M; k++) {
    h_rl[k] = h_cf[2*k];
    h_im[k] = h_cf[2*k+1];
  }

  if(verbose) {
    for(c=0; c<M; c++) {
      printf("\n GPU Column %lu:", (unsigned long)c);
      for(k=0; k<N; k+=10) {
	for(j=k; j<(N<(k+10)?N:k+10); j++)
	  printf("%s%7.4f", (j%10)?" ":"\nReal: ", h_rl[c*N+j]);
	for(j=k; j<(N<(k+10)?N:k+10); j++)
	  printf("%s%7.4f", (j%10)?" ":"\nImag: ", h_im[c*N+j]);
      }
    }
    printf("\n");
  }

  printf("**** %s unsort: GPU-CPU max real diff: %g\n", type,
	 max_abs_vec_diff(N*M, rl, h_rl));

  printf("**** %s unsort: GPU-CPU max imag diff: %g\n", type,
	 max_abs_vec_diff(N*M, im, h_im));

  // Sort tests
  Float *srt = (Float *) malloc(N*M*sizeof(Float));

  c_real_dft_sort(N, M, rl, im, srt);

  if(verbose) {
    printf("CPU sorted vec:");
    for(c=0; c<M; c++) {
      printf("\n Column %lu:", (unsigned long)c);
      for(k=0; k<N; k++)
	printf("%s%7.4f", (k%10)?" ":"\n", srt[c*N+k]);
    }
    printf("\n");
  }

  Float *d_srt;
  gpuErrChk(cudaMalloc(&d_srt, N*M*sizeof(Float)),
	    "cuda_real_dft_sort_tst:cuda_error", "");
  Float *h_srt = (Float *) malloc( N*M*sizeof(Float));

  h_real_dft_sort(N, M, d_cf, d_srt);
  gpuErrChk(cudaMemcpy(h_srt, d_srt, N*M*sizeof(Float), cudaMemcpyDeviceToHost),
	    "cuda_real_dft_sort_tst:cuda_error", "");
  
  if(verbose) {
    printf("GPU sorted vec:");
    for(c=0; c<M; c++) {
      printf("\n Column %lu:", (unsigned long)c);
      for(k=0; k<N; k++)
	printf("%s%7.4f", (k%10)?" ":"\n", h_srt[c*N+k]);
    }
    printf("\n");
  }

  printf("**** %s sort: GPU-CPU max diff: %g\n", type,
	 max_abs_vec_diff(N*M, srt, h_srt));

  printf("**** %s CPU sort-unsort diff with original: %g\n",
	 type, max_abs_vec_diff(N*M, vec, srt));

  printf("**** %s GPU sort-unsort diff with original: %g\n",
	 type, max_abs_vec_diff(N*M, vec, h_srt));

  free(h_srt);
  gpuErrChk(cudaFree(d_srt),
	    "cuda_real_dft_sort_tst:cuda_error", "");
  free(srt);
  free(h_im);
  free(h_rl);
  free(h_cf);
  gpuErrChk(cudaFree(d_vec),
	    "cuda_real_dft_sort_tst:cuda_error", "");
  free(im);
  free(rl);
}

int main(int argc , char *argv[]) {
  int err = EXIT_SUCCESS;
  size_t N,M,k;
  unsigned long nv,mv;

  if(argc<3 || argc>4)
    help(argv[0]);

  verbose = !strcmp(argv[1],"-v");

  if(argc-verbose != 3 || (sscanf(argv[verbose+1],"%lu",&nv) != 1) ||
     (sscanf(argv[verbose+2],"%lu",&mv) != 1) ||
     (nv == 0) || (mv == 0))
    help(argv[0]);

  N = nv;
  M = mv;

  double * dbl = (double *)malloc(N*M*sizeof(double));
  for(k=0; k<N*M; k++)
    dbl[k] = double(rand()-(RAND_MAX/2))/double(RAND_MAX);

  run_tests(N, M, dbl, "double float");

  float * flt = (float *)malloc(N*M*sizeof(float));
  for(k=0; k<N*M; k++)
    flt[k] = (float)dbl[k];

  run_tests(N, M, flt, "single float");

  free(flt);
  free(dbl);

  return err;
}
