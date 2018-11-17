#ifndef __cumsum_HDR__
#define __cumsum_HDR__

/** \file
Function to compute the cumulative sum of a vector. The cumulative sum of x[0],...x[n]
is defined by:
   y[k] = x[0]+x[1]+...x[k], k=0,...,n-1
*/

#ifndef __HOST_DEVICE__
#ifdef __CUDA_ARCH__
#define __HOST_DEVICE__ __host__ __device__
#else
#define __HOST_DEVICE__
#endif
#endif

template <class T>
__HOST_DEVICE__ void c_cumsum(size_t n_vec, //!< Vector length
			      const T *vec,	    //!< input
			      T *csum	// 1< Output. can be same as vec.
			      )
{
  T sum = 0;
  for(size_t k=0; k<n_vec; k++) 
    csum[k] = (sum += vec[k]);
}

#ifdef __NVCC__

template <class T>
__global__ void
d_cumsum(size_t n_vec,
	 int bgn_shft,
	 int end_shft,
	 T *vec
	 )
{
  size_t indx = blockIdx.x * blockDim.x + threadIdx.x;

  for(int shft=bgn_shft; shft<=end_shft; shft++) {
    size_t grp = indx >> shft;
    size_t grp_size = 1<<shft;
    size_t base = grp_size * (2*grp+1);
    size_t i = base + (indx & (grp_size-1));
    if(i>=n_vec)
      break;

    vec[i] += vec[base-1];
    
    __syncthreads();
  }
}

template <class T>
__host__ void h_cumsum(size_t n_vec,	/**< no. of elements in the block */
		       T *res	//!< input and output of size \c n_vec.
		       )
{
  const size_t n_thrds = n_vec>>1;
  if(n_thrds == 0) return;

  const int log2_blk = cuda_dev->log2ThrdsBlk();
  const size_t max_thrds_blk = 1<<log2_blk;
  const size_t n_thrds_per_blk = n_thrds<max_thrds_blk? n_thrds: max_thrds_blk;
  const size_t n_blks = (n_thrds+max_thrds_blk-1) >> log2_blk;
  int shft;

  d_cumsum <<< n_blks, n_thrds_per_blk >>> (n_vec, 0, log2_blk, res);
  cudaDeviceSynchronize();

  for(shft=log2_blk+1; (n_vec-1)>>shft > 0; shft++) {
    d_cumsum <<< n_blks, n_thrds_per_blk >>> (n_vec, shft, shft, res);
    cudaDeviceSynchronize();
  }
}

template <class T>
__host__ void h_cumsum(size_t n_vec,	/**< no. of elements in the block */
		       const T *vec,     /**< the source vector of length \c n_vec.
			       if NULL assume it is already copied
			       to res*/
		       T *res			// Output vector of size \c n_vec.
		       )
{
  if(vec != NULL)
    gpuErrChk(cudaMemcpy(res, vec, n_vec*sizeof(T),cudaMemcpyDeviceToDevice),
	      "h_cumsum:Memcpy","");

  h_cumsum(n_vec, res);
}
#endif

#endif	/* __cumsum_HDR__ */
