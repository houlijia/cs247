#ifndef __CUDA_SUB_SQR_HDR__
#define __CUDA_SUB_SQR_HDR__
/**
   Subtract a constant from an array and then square the elements of the array:
     res = (vec-sbval)^2. Return result in res.
  */

#include <stddef.h>

#include "CudaDevInfo.h"

template <class T>
__global__ void 
d_sub_sqr(const T sbval,	//!< pointer to value to subtract
	  const T *vec,       //!< vector
	  size_t n_vec,	//!< No. of elements in \c vec
	  T *res	//!< Result vector
	  )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec) {
    T dff = vec[i] - sbval;
    res[i] = dff * dff;
  }
}

template <class T>
void h_sub_sqr(const T sbval,	//!< Value to subtract
	       const T *vec,       //!< vector
	       size_t n_vec,	//!< No. of elements in \c vec
	       T *res	//!< Result vector
	)
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_sub_sqr<T> <<< n_blks, n_thrds_per_blk >>> (sbval, vec, n_vec, res);

  cudaDeviceSynchronize();
}

template <class T>
__global__ void 
d_sub_sqr(const T *sbval,	//!< pointer to value to subtract
	  const T *vec,       //!< vector
	  size_t n_vec,	//!< No. of elements in \c vec
	  T *res	//!< Result vector
	  )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec) {
    T dff = vec[i] - *sbval;
    res[i] = dff * dff;
  }
}

template <class T>
void h_sub_sqr(const T *sbval,	//!< Value to subtract
	       const T *vec,       //!< vector
	       size_t n_vec,	//!< No. of elements in \c vec
	       T *res	//!< Result vector
	)
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_sub_sqr<T> <<< n_blks, n_thrds_per_blk >>> (sbval, vec, n_vec, res);

  cudaDeviceSynchronize();
}

#endif /*  __CUDA_SUB_SQR_HDR__ */
