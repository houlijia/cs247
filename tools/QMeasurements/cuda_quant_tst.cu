#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "CudaDevInfo.h"

#include "cuda_quant.h"
#include "cc_quant.h"

typedef int QV;
typedef short LBL;

template <class T>
T max_abs_vec_diff(size_t n_vec,	//!< no of elements
		   const T *vec1,	//!< The vector
		   const T *vec2	//!< The vector
	      ) {
  T mx = fabs(double(vec1[0]) - double(vec2[0]));
  size_t i;
  
  for(i=1; i<n_vec; i++) {
    T df = (T) fabs(double(vec1[i]) - double(vec2[i]));
    if(df > mx)
      mx = df;
  }

  return mx;
}

template <class FLT> 
static void run_tst(int verbose,
		    const char *tst_name,
		    size_t n_vec,
		    FLT *vec,
		    FLT intvl,
		    FLT offset,
		    LBL sat_lvl,
		    size_t nnclp,
		    int do_save
		    )
{
  printf("\n *** %s Test ***\n"
	 "n_vec=%lu nnclp=%lu sat_lvl=%lu do_save=%d\n"
	 "intvl=%lg offset=%lg\n",
	 tst_name, (unsigned long) n_vec, (unsigned long) nnclp,
	 (unsigned long) sat_lvl, do_save,
	 (double) intvl, (double) offset);

  FLT *d_vec;
  gpuErrChk(cudaMalloc(&d_vec, n_vec*sizeof(FLT)), "cuda_quant_tst:cuda_error", "");
  gpuErrChk(cudaMemcpy(d_vec, vec, n_vec*sizeof(FLT),cudaMemcpyHostToDevice), "cuda_quant_tst:cuda_error", "");

  QV *cpu_no_clip = (QV *)malloc(nnclp*sizeof(QV));
  QV *d_gpu_no_clip;
  gpuErrChk(cudaMalloc(&d_gpu_no_clip, nnclp*sizeof(QV)), "cuda_quant_tst:cuda_error", "");
  QV *h_gpu_no_clip= (QV *)malloc(nnclp*sizeof(QV));
 
  LBL *cpu_q_vec = (LBL *)malloc((n_vec-nnclp)*sizeof(LBL));
  LBL *d_gpu_q_vec;
  gpuErrChk(cudaMalloc(&d_gpu_q_vec, (n_vec-nnclp)*sizeof(LBL)), "cuda_quant_tst:cuda_error", "");
  LBL *h_gpu_q_vec = (LBL *)malloc((n_vec-nnclp)*sizeof(LBL));

  QV *cpu_save, *h_gpu_save, *d_gpu_save;
  size_t cpu_n_save, h_gpu_n_save, *d_gpu_n_save;

  if(do_save) {
    cpu_save = (QV *)malloc((n_vec-nnclp)*sizeof(QV));  
    gpuErrChk(cudaMalloc(&d_gpu_save, (n_vec-nnclp)*sizeof(QV)), "cuda_quant_tst:cuda_error", "");
    h_gpu_save = (QV *)malloc((n_vec-nnclp)*sizeof(QV));
    gpuErrChk(cudaMalloc(&d_gpu_n_save, sizeof(size_t)), "cuda_quant_tst:cuda_error", "");
  }
  else {
    cpu_save = NULL;
    d_gpu_save = NULL;
    h_gpu_save = NULL;
    d_gpu_n_save = NULL;
  }

  cpu_n_save = c_quant(nnclp, n_vec-nnclp, vec, vec+nnclp, intvl, offset, sat_lvl,
		       cpu_no_clip, cpu_q_vec, cpu_save);

  h_quant(nnclp, n_vec-nnclp, d_vec, d_vec+nnclp, intvl, offset, sat_lvl, 
	  d_gpu_no_clip, d_gpu_q_vec, d_gpu_save, d_gpu_n_save);

  gpuErrChk(cudaFree(d_vec), "cuda_quant_tst:cuda_error", "");

  cudaDeviceSynchronize();
  gpuErrChk(cudaGetLastError(), "cuda_quant_tst:cuda_error", "");

  // Copy GPU results to CPU and release GPU buffers
  gpuErrChk(cudaMemcpy(h_gpu_no_clip, d_gpu_no_clip, nnclp*sizeof(QV),
		       cudaMemcpyDeviceToHost),"cuda_quant_tst:cuda_error", "");
  gpuErrChk(cudaFree(d_gpu_no_clip),"cuda_quant_tst:cuda_error", "");
  gpuErrChk(cudaMemcpy(h_gpu_q_vec, d_gpu_q_vec,
		       (n_vec-nnclp)*sizeof(LBL), cudaMemcpyDeviceToHost),"cuda_quant_tst:cuda_error", "");
  gpuErrChk(cudaFree(d_gpu_q_vec),"cuda_quant_tst:cuda_error", "");

  if(do_save) {
    gpuErrChk(cudaMemcpy(&h_gpu_n_save, d_gpu_n_save, sizeof(size_t),
			 cudaMemcpyDeviceToHost),"cuda_quant_tst:cuda_error", "");
    gpuErrChk(cudaFree(d_gpu_n_save), "cuda_quant_tst:cuda_error", "");
    gpuErrChk(cudaMemcpy(h_gpu_save, d_gpu_save, h_gpu_n_save*sizeof(QV),
			 cudaMemcpyDeviceToHost),"cuda_quant_tst:cuda_error", "");
    gpuErrChk(cudaFree(d_gpu_save),"cuda_quant_tst:cuda_error", "");
  }

  if(verbose) {
    printf("--------------\n");
    size_t k,j;
    for(k=0; k<n_vec; k+=10) {
      size_t k_end = (k+10)<n_vec? k+10: n_vec;

      printf("\n Ind:");
      for(j=k; j<k_end; j++)
	printf(" %6lu", (unsigned long)j);

      printf("\n vec:");
      for(j=k; j<k_end; j++)
	printf(" %6.1f", vec[j]);

      printf("\n CPU:");
      for(j=k; j<k_end; j++) {
	if(j<nnclp)
	  printf(" %6d", (int) cpu_no_clip[j]);
	else
	  printf(" %6d", (int) cpu_q_vec[j-nnclp]);
      }

      printf("\n GPU:");
      for(j=k; j<k_end; j++) {
	if(j<nnclp)
	  printf(" %6d", (int) h_gpu_no_clip[j]);
	else
	  printf(" %6d", (int) h_gpu_q_vec[j-nnclp]);
      }
    }
    
    if(do_save) {
      printf("\n CPU Save:");
      for(k=0; k<cpu_n_save; k++)
	printf("%s%7d", (k%10)?" ":"\n", cpu_save[k]);
      printf("\n GPU Save:");
      for(k=0; k<h_gpu_n_save; k++)
	printf("%s%7d", (k%10)?" ":"\n", h_gpu_save[k]);
    }

    printf("\n--------------\n");
  }

  if(nnclp)
    printf("no_clip compare: %d\n",
	   (int) max_abs_vec_diff(nnclp, cpu_no_clip, h_gpu_no_clip));
	   
  if(n_vec > nnclp)
    printf("q_vec compare: %d\n",
	   (int) max_abs_vec_diff(n_vec-nnclp, cpu_q_vec, h_gpu_q_vec));

  if(do_save) {
    long n_save_diff = (long)cpu_n_save - (long)h_gpu_n_save;
    printf("n_save compare: %ld\n", n_save_diff);

    if(!n_save_diff) {
      if(cpu_n_save)
	printf("save compare: %lu\n",
	       (unsigned long) max_abs_vec_diff(cpu_n_save, cpu_save, h_gpu_save));
      else
	printf("no save values\n");
    }
  }

  printf("==========================\n");

  free(cpu_no_clip);
  free(h_gpu_no_clip);
  free(cpu_q_vec);
  free(h_gpu_q_vec);
  free(cpu_save);
  free(h_gpu_save);
}
	     
static void help(const char *cmd)
{
  fprintf
    (stderr, "USAGE:\n"
     "    %s [-v] <n_vec> <intvl> <offset> <sat_lvl> <n_no_clip> <do_save>\n"
     "The program generates a random vector of <n_vec> entries between -100 and 100\n"
     "and quantize it using c_quant and h_cuda_quant and compares the results.\n"
     "If -v is specified the input and output vectors are printred.\n"
     "<n_vec> is a non-negative integer\n"
     "<intvl> is a non-negative real\n"
     "<offset> is a real\n"
     "<sat_lvl> is a positive integer\n"
     "<n_no_clip> is a non-negative integer\n"
     "<do_save> is 0 or 1 (logical integer\n",
     cmd);
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  unsigned int nv, nnclp, do_save;
  LBL sat_lvl;
  double dbl_intvl, dbl_offset;
  
  if(argc < 7)
    help(argv[0]);


  int  verbose = !strcmp(argv[1],"-v");

  if(argc-verbose != 7 ||
     (sscanf(argv[verbose+1],"%u"  ,&nv) + 
      sscanf(argv[verbose+2],"%lg", &dbl_intvl) +
      sscanf(argv[verbose+3],"%lg", &dbl_offset) +
      sscanf(argv[verbose+4],"%hu", &sat_lvl) +
      sscanf(argv[verbose+5],"%u",  &nnclp) +
      sscanf(argv[verbose+6],"%u",  &do_save)) != 6 ||
     dbl_intvl <= 0 || sat_lvl == 0
     )
    help(argv[0]);

  float flt_intvl = (float) dbl_intvl;
  float flt_offset = (float) dbl_offset;

  // Memory allocation and initialize random data
  const size_t n_vec = nv;
  double *dbl = (double *) malloc(n_vec*sizeof(double)); 
  float *flt = (float *) malloc(n_vec*sizeof(float)); 

  // Initialize random data
  size_t i;
  for(i=0; i<n_vec; i++) {
    dbl[i] = -100. + (200./double(RAND_MAX))*double(rand());
    flt[i] = float(dbl[i]);
  }

  run_tst(verbose, "double", n_vec, dbl, dbl_intvl, dbl_offset, sat_lvl, nnclp, do_save);
  run_tst(verbose, "single", n_vec, flt, flt_intvl, flt_offset, sat_lvl, nnclp, do_save);

  free(flt);
  free(dbl);
}
