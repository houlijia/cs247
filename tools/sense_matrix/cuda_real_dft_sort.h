#ifndef __CUDA_REAL_DFT_SORT_HDR__
#define __CUDA_REAL_DFT_SORT_HDR__

/**
   \file

   function to convert between DFT output of a real signal of a more compact
   form. Let \c N be the DFT order and let x[0],...,x[N-1] be the complex DFT
   coefficients. If N is even, the compact form is:

   Re{x[0]},Re{x[N/2]},Re{x[1]},Im{x[1]},...,Re{x[N/2-1]},Im{x[N/2-1]}

   If N>1 is odd the compact form is 

   Re{x[0]},Re{x[1]},Im{x[1]},...,Re{x[(N-1)/2]},Im{x[(N-1)/2]}

   If N=1 the compact form is

   Re{x[0]}

   The functions perform the conversions on the columns of matrices with M
   columns.

   The complex DFT coefficients (in the non compact form) are organized
   sequentially as pairs of (Re,Im).

   The input and output must be separate (the conversion cannot be in place).

*/

#include <stddef.h>
#include <math.h>
#include <stdio.h>

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define printf mexPrintf
#endif

#include "CudaDevInfo.h"
#include "cuda_flt_funcs.h"

#ifndef ulong
#define ulong (unsigned long)
#endif

template <class Float>
__global__ void d_real_dft_sort(size_t N,	//!< DFT order (>0)
				size_t M,  //!< Number of columns (>0)
				const Float *cf, //!< DFT coefficients size 2*M*N
				Float *cmpct     //!< Output array of size M*N
				)
{
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const Float *cf_cl = cf + blockIdx.y * N * 2;
  Float *cmpct_cl = cmpct + blockIdx.y * N;

  if(N%2) {
    if(j==0)
      cmpct_cl[j] = cf_cl[j];
    else if(j<N)
      cmpct_cl[j] = cf_cl[j+1];
  }
  else {
    if(j==1)
      cmpct_cl[j] = cf_cl[N];
    else if(j<N)
      cmpct_cl[j] = cf_cl[j];
  }
}

template <class Float>
__host__ void h_real_dft_sort(size_t N,	//!< DFT order (>0)
			      size_t M,  //!< Number of columns (>0)
			      const Float *cf, //!< DFT coefficients size 2*M*N
			      Float *cmpct     //!< Output array of size M*N
			      )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;

  dim3 n_blks ((N+max_thr_blk-1)/max_thr_blk, M);
  int n_thrds_per_blk = (N<max_thr_blk)? N: max_thr_blk;

  d_real_dft_sort <<< n_blks, n_thrds_per_blk >>>(N, M, cf, cmpct);
  cudaStreamSynchronize(0);
}

template <class Float>
__global__ void d_real_dft_unsort(size_t N, //!< DFT order (>0)
				  size_t M, //!< Number of columns (>0)
				  const Float *cmpct, //!< input array of size M*N
				  Float *cf //!< output DFT coefficients size 2*M*N
				  )
{
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const Float *cmpct_cl = cmpct + blockIdx.y * N;
  Float *cf_cl = cf + blockIdx.y * N * 2;

  if(N%2) {
    if(j==0) {
      cf_cl[0] = cmpct_cl[0];
      cf_cl[1] = Float(0);
    }
    else if(j<=N/2) {
      cf_cl[2*j] = cmpct_cl[2*j-1];
      cf_cl[2*j+1] = cmpct_cl[2*j];
    }
    else if(j<N) {
      cf_cl[2*j] = cmpct_cl[2*(N-j)-1];
      cf_cl[2*j+1] = -cmpct_cl[2*(N-j)];
    }
  }
  else {
    if(j==0) {
      cf_cl[0] = cmpct_cl[0];
      cf_cl[1] = Float(0);
    }
    else if(j==N/2) {
      cf_cl[N] = cmpct_cl[1];
      cf_cl[N+1] = Float(0);
    }
    else if(j<N/2) {
      cf_cl[2*j] = cmpct_cl[2*j];
      cf_cl[2*j+1] = cmpct_cl[2*j+1];
    }
    else if(j<N) {
      cf_cl[2*j] = cmpct_cl[2*(N-j)];
      cf_cl[2*j+1] = -cmpct_cl[2*(N-j)+1];
    }
  }
}

template <class Float>
__host__ void h_real_dft_unsort(size_t N, //!< DFT order (>0)
				size_t M, //!< Number of columns (>0)
				const Float *cmpct, //!< input array of size M*N
				Float *cf //!< output DFT coefficients size 2*M*N
				)
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;

  dim3 n_blks ((N+max_thr_blk-1)/max_thr_blk, M);
  int n_thrds_per_blk = (N<max_thr_blk)? N: max_thr_blk;

  d_real_dft_unsort <<< n_blks, n_thrds_per_blk >>>(N, M, cmpct, cf);
  cudaStreamSynchronize(0);
}

#endif	/* __CUDA_REAL_DFT_SORT_HDR__ */
