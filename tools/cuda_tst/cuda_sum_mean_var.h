#ifndef __CUDA_SUM_MEAN_VAR_HDR__
#define __CUDA_SUM_MEAN_VAR_HDR__

/** \file
Compute sum, mean, var over blocks. Note that the vector is assumed to have
at least one elemement.
*/

#include <stddef.h>
#include <math.h>

#include "mex_assert.h"
#include "mex_tools.h"
#include "CudaDevInfo.h"
#include "cuda_vec_op_scalar.h"
#include "cuda_vec_op_unary.h"
#include "cuda_sub_sqr.h"

/** Given a vector, sum up the elements in each block into the first element of
    the block. The rest of the elements in the block are destroyed.
*/
template <class T>
__global__ void
d_sum_vec_blk(size_t n_vec,	//!< no. of elements in the block
	      size_t stride,	//!< step between adjacent element in the block
	                        //!< (1 for normal vectors)
	      T* vec		//!< The vector
	      )
{
  const size_t bsize = (gridDim.x==1)? n_vec: blockDim.x*2;
  const size_t bbgn = blockIdx.x * bsize;
  size_t blen = (n_vec - bbgn > bsize)? bsize: (n_vec - bbgn);

  while(blen > 1) {
    size_t cnt = blen/2;
    size_t j = threadIdx.x + blen - cnt;
    if(j >= blen)
      return;

    vec[(bbgn+threadIdx.x)*stride] += vec[(bbgn+j)*stride];

    blen = blen - cnt;

    __syncthreads();
  }
}

/** Compute the sum of a vector of length > 0
 */
template <class T>
__host__ void
h_sum_vec(size_t n_vec,	/**< no. of elements in the block */
	  const T *vec,     /**< the source vector of length \c n_vec.
			       May be equal to \c res. */
	  T *res           /**< vector of length \c n_vec. result is 
			      returned in \c res[0]. The rest is
			      scratch. If NULL assume that the input is
			      already in \c res */
	  )
{
  if(vec != (const T*)res)
    gpuErrChk(cudaMemcpy(res, vec, n_vec*sizeof(T),cudaMemcpyDeviceToDevice),
	      "h_sum_vec:memcpy", "");

  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
  const size_t bsize = 2*max_thr_blk;
  size_t stride = 1;

  while(n_vec > 1) {
    /* Do the summation over blocks of size max_thr_blk*2, where at the end
       the sum of the block is in the first entry of the block. Then sum up
       the first entries using the same methods but with stride multiplied by
       max_thr_blk*2.
    */
    size_t n_blks = (n_vec+bsize-1)/bsize;
    size_t n_thrds_per_blk = (n_vec <bsize)? (n_vec/2):max_thr_blk;

    d_sum_vec_blk <<< n_blks, n_thrds_per_blk >>> (n_vec, stride, res);

    n_vec = n_blks;
    stride *= bsize;

    cudaDeviceSynchronize();

  }
}

/** Compute the mean of a vector of length > 0
 */
template <class T>
__host__ void
h_mean_vec(size_t n_vec,	/**< no. of elements in the block */
	  const T *vec,		/**< the source vector of length \c n_vec.
				   May be equal to \c res. */
	  T *res		/**< vector of length \c n_vec. result is 
				   returned in \c res[0]. The rest is
				   scratch. If NULL assume that the input is
				   already in \c res */
	  )
{
  h_sum_vec(n_vec, vec, res);
  d_vec_div_scalar <<< 1,1 >>> (T(n_vec), res, 1, res);
  cudaDeviceSynchronize();
}
  
/** Compute the standard deviation of a vector of length > 1
 */
template <class T>
__host__ void
h_stdv_vec(size_t n_vec,	/**< no. of elements in the block */
	   const T *vec,     /**< the source vector of length \c n_vec.
				May be equal to \c res. */
	   const T *pmean,	/**<< The mean of the vector (in GPU) */
	   T *res            /**< vector of length \c n_vec. result is 
				returned in \c res[0]. The rest is
				scratch. */
	   )
{
  h_sub_sqr(pmean, vec, n_vec, res);
  h_sum_vec(n_vec, res, res);
  h_vec_div_scalar (T(n_vec)-T(1), res, 1, res);
  h_vec_sqrt (res, 1, res);
}
  
/** Compute mean ad standard deviation of a vector of length > 1
 */
template <class T>
__host__ void
h_mean_stdv_vec(size_t n_vec,	/**< no. of elements in the block */
	   const T *vec,     /**< the source vector of length \c n_vec.
				May NOT be equal to \c res. */
	   T *res            /**< vector of length \c n_vec+1. mean is
				returned in \c res[0] and standard deviation
				is returned in \c res[1]. The rest is scratch. */
	   )
{
  h_mean_vec(n_vec, vec, res);
  h_stdv_vec(n_vec, vec, res, res+1);
}
  
#endif	/*  __CUDA_SUM_MEAN_VAR_HDR__ */
