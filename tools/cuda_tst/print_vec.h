#ifndef __print_vec_HDR__
#define __print_vec_HDR__

/** \file
 */

#include <assert.h>
#include <stdio.h>

#include "CudaDevInfo.h"
#include "fast_heap.h"

template <class T> __HOST_DEVICE__ void
c_print_vec(size_t sz,	//!< number of elements in vector
	    const T vec[],	//!< The vector
	    size_t n_row,	//!< Number of elements to print in a row
	    const char *fmt	//!< Format for a single element (including
	    )
{
  size_t j,k;
  for(j=0; j<sz; j+=n_row) {
    printf("%5lu: ", (unsigned long)j);

    for(k=j; k<j+n_row && k<sz; k++)
      printf(fmt, vec[k]);
    printf("\n");
  }
}

#ifdef __NVCC__

template <class T> __global__  void
d_print_vec(size_t sz,	//!< number of elements in vector
	    const T vec[],	//!< The vector
	    size_t n_row,	//!< Number of elements to print in a row
	    const char *fmt	//!< Format for a single element (including
	    )
{
  c_print_vec(sz, vec,n_row, fmt);
}

template <class T> __host__  void
h_print_vec(size_t sz,	//!< number of elements in vector
	    const T vec[],	//!< The vector
	    size_t n_row,	//!< Number of elements to print in a row
	    const char *fmt	//!< Format for a single element (including
	    )
{
  size_t sz_fmt = strlen(fmt)+1;
  GenericHeapElement &pd_fmt = d_fast_heap->get(sz_fmt);
  char *d_fmt = static_cast<char*>(*pd_fmt);

  gpuErrChk(cudaMalloc(&d_fmt, sz_fmt), "h_print_vec:cudaMalloc", "");
  gpuErrChk(cudaMemcpy(d_fmt, fmt, sz_fmt, cudaMemcpyHostToDevice),
	    "h_print_vec:cudaMemcpy", "");

  d_print_vec <<< 1,1 >>> (sz, vec,n_row, d_fmt);
  cudaStreamSynchronize(0);

  pd_fmt.discard();
}


#endif

template <class T>
void print_vec(size_t sz,	//!< number of elements in vector
	       const T vec[],	//!< The vector
	       size_t n_row,	//!< Number of elements to print in a row
	       const char *fmt,	//!< Format for a single element (including
				//! leading spaces
	       bool on_gpu=false
	       )
{
#if defined(__NVCC__)
  if(on_gpu) {
    h_print_vec(sz, vec, n_row, fmt);
    return;
  }
#else
  (void) on_gpu;
  assert(!on_gpu);
#endif
  
  c_print_vec(sz, vec, n_row, fmt);
}

#endif	/*  __print_vec_HDR__ */
