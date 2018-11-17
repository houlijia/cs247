#ifndef __CUDA_VEC_OP_UNARY_HDR__
#define __CUDA_VEC_OP_UNARY_HDR__

/** \file
Functions to perform unary operations on vectors, entry by entry.
*/

#include <stddef.h>

#include "CudaDevInfo.h"

/** take square root of a vector \c vec of length \c n_vec and return results in res
    (res can be the same as vec) */
template <class T>
__global__ void
d_vec_sqrt(const T *vec,
	   size_t n_vec,
	   T *res
	   )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec)
    res[i] = T(sqrt(vec[i]));
}

/** take square root of a vector \c vec of length \c n_vec and return results in res
    (res can be the same as vec) */
template <class T>
__host__ void
h_vec_sqrt(const T *vec,
	   size_t n_vec,
	   T *res
	   )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_sqrt <<< n_blks, n_thrds_per_blk >>> (vec, n_vec, res);

  cudaDeviceSynchronize();
}

/** take absolute value of a vector \c vec of length \c n_vec and return results in res
    (res can be the same as vec) */
template <class T>
__global__ void
d_vec_abs(const T *vec,
	  size_t n_vec,
	  T *res
	  )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec)
    res[i] = (vec[i]<0)? -vec[i]: vec[i];
}

/** take absolute value of a vector \c vec of length \c n_vec and return results in res
    (res can be the same as vec) */
template <class T>
__host__ void
h_vec_abs(const T *vec,
	  size_t n_vec,
	  T *res
	  )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_abs <<< n_blks, n_thrds_per_blk >>> (vec, n_vec, res);

  cudaDeviceSynchronize();
}

#endif	/*  __CUDA_VEC_OP_UNARY_HDR__ */
