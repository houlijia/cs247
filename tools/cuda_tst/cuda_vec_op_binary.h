#ifndef __cuda_vec_op_binary_HDR__
#define __cuda_vec_op_binary_HDR__

/** \file
Template function to perform element-wise binary operations on vectors. The different
argument vectors are epxected to be either disjoint or identical, but not paritally
overlapping.
*/

//! x -= y
template <class T>
__global__ void
d_sub_asgn(size_t n_vec,
	   T *x,
	   const T *y
	   )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;

  if(indx < n_vec)
    x[indx] -= y[indx];
}


template <class T>
__host__ void
h_sub_asgn(size_t n_vec,	/**< no. of elements in the block */
	   T *x,
	   const T *y
	   )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;
	      
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_sub_asgn <<< n_blks, n_thrds_per_blk >>> (n_vec, x, y);

  cudaDeviceSynchronize();
}

#endif	/* __cuda_vec_op_binary_HDR__ */
