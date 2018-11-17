#ifndef __CUDA_VEC_OP_SCALAR_HDR__
#define __CUDA_VEC_OP_SCALAR_HDR__

/** \file
Functions to perform operations between a vector and a scalar. The operations
are performed between each entry of the vector and the scalar and the results
are returned in the same vector. The scalar may be supplied as a pointer to a
value on the GPU or as an argument.
*/

#ifdef __NVCC__

#include <stddef.h>

#include "CudaDevInfo.h"

/** Divide a vector \c vec of length \c n_vec by \c scalar and
 return result in \c res
*/
template <class T>
__global__ void
d_vec_div_scalar(T scalar,
		 const T *vec,
		 size_t n_vec,
		 T *res
		 )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec)
    res[i] = vec[i] / scalar;
}

/** Multiply a vector \c vec of length \c n_vec by \c scalar and
 return result in \c res
*/
template <class T>
__global__ void
d_vec_mlt_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T *res
		 )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec)
    res[i] = vec[i] * scalar;
}

/** Add a scalar \c scalar to a vector \c vec of length \c n_vec and
 return result in \c res
*/
template <class T>
__global__ void
d_vec_add_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T *res
		 )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec)
    res[i] = vec[i] + scalar;
}

/** Subtract a scalar \c scalar from a vector \c vec of length \c n_vec and
 return result in \c res
*/
template <class T>
__global__ void
d_vec_sub_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T *res
		 )
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n_vec)
    res[i] = vec[i] - scalar;
}

/** Divide a vector \c vec of length \c n_vec by \c scalar and
 return result in \c res
*/
template <class T>
__host__ void
h_vec_div_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_div_scalar <<< n_blks, n_thrds_per_blk >>> (scalar, vec, n_vec, res);

  cudaDeviceSynchronize();
}

/** Divide a vector \c vec of length \c n_vec by \c scalar and
 return result in place
*/
template <class T>
__host__ void
h_vec_div_scalar(T scalar,
		 size_t n_vec,
		 T* vec
		 )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_div_scalar <<< n_blks, n_thrds_per_blk >>> (scalar, vec, n_vec, vec);

  cudaDeviceSynchronize();
}

/** Multiply a vector \c vec of length \c n_vec by \c scalar and
 return result in \c res
*/
template <class T>
__host__ void
h_vec_mlt_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_mlt_scalar <<< n_blks, n_thrds_per_blk >>> (scalar, vec, n_vec, res);

  cudaDeviceSynchronize();
}

/** Add a scalar \c scalar to a vector \c vec of length \c n_vec and
 return result in \c res
*/
template <class T>
__host__ void
h_vec_add_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_add_scalar <<< n_blks, n_thrds_per_blk >>> (scalar, vec, n_vec, res);

  cudaDeviceSynchronize();
}

/** Subtract a scalar \c scalar from a vector \c vec of length \c n_vec and
 return result in \c res
*/
template <class T>
__host__ void
h_vec_sub_scalar(T scalar,
		 const T*vec,
		 size_t n_vec,
		 T* res
		 )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_vec_sub_scalar <<< n_blks, n_thrds_per_blk >>> (scalar, vec, n_vec, res);

  cudaDeviceSynchronize();
}

#endif	/* #ifdef __NVCC__ */
#endif	/*  __CUDA_VEC_OP_SCALAR_HDR__ */
