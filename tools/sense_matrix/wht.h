#ifndef __wht_h__
#define __wht_h__

/** \file
Template unctions to compute Fast Walsh Hadamard transform and inverse transform
*/


#include "CudaDevInfo.h"
#include "cc_vec_op_scalar.h"
#include "cuda_vec_op_scalar.h"

//* Perofrm operation on one pair
template <typename T>
__HOST_DEVICE__ void do_wht_step(T &b0, T &b1)
{
  T v0 = b0 + b1;
  T v1 = b0 - b1;
  b0 = v0;
  b1 = v1;
}
  

//* Perform in place, one step of WHT
template <typename T>
__HOST_DEVICE__  void wht_step(T *vec, //*< Input and output vector
			       size_t veclen, //*< length of vec
			       size_t step  //*< step betwee b0 and b1
			      )
{
  const size_t blk_size = step << 1;

  for(size_t blk_bgn=0; blk_bgn<veclen; blk_bgn += blk_size) {
    T *b0 = vec + blk_bgn;
    T *b1 = b0 + step;
    for(size_t k=0; k<step; k++) {
      do_wht_step(b0[k], b1[k]);
    }
  }
}

//* Template for integer types - check if a number is a power of 2
template <typename INT>
bool is_pwr_of_2(INT val)
{
  if(val <= 0) return false;
  for(; val > 1; val >>= 1)
    if(val & INT(1)) return false;
  return true;
}

//* Compute inverse WHT of a matrix (Hadamard order), column by column, that
//* is compute the product W*M, where W is a WH matrix and M is the given
//* matrix. Result is the same matrix.
template <typename T>
void c_iwht(T *mtx,		//*< Matrix to be multiplied (in place)
	    size_t nr,		//*< Number of rows in mtx (must be a power of 2)
	    size_t nc		//*< Number of columns in mtx
	    )
{
  for(size_t step=1; step<nr; step *= 2)
    wht_step(mtx, nr*nc, step);
}
	   
//* Compute WHT of a matrix (Hadamard order), column by column, that is
//* compute the product W*M, where W is a WH matrix and M is the given matrix,
//* and then divide the result by the number of rows in W. Result is the same
//* matrix.
template <typename T>
void c_wht(T *mtx,		//*< Matrix to be multiplied (in place)
	   size_t nr,		//*< Number of rows in mtx (must be a power of 2)
	   size_t nc		//*< Number of columns in mtx
	   )
{
  c_iwht(mtx, nr, nc);
  c_vec_div_scalar(T(nr), nr*nc, mtx);
}

#ifdef __NVCC__

template <typename T>
__global__ void d_iwht0(T *mtx)
{
  mtx += 2*blockDim.x*(blockIdx.x + gridDim.x*blockIdx.y);
  for(size_t step=1; step <= blockDim.x; step *= 2) {
    T *b0 = mtx + threadIdx.x%step + 2*step*(threadIdx.x/step);
    do_wht_step(b0[0], b0[step]);
    __syncthreads();
  }
}

template <typename T>
__global__ void d_iwht1(T *mtx)
{
  size_t step = blockDim.x * gridDim.x;
  T *b0 = mtx + threadIdx.x + 
    blockDim.x*(blockIdx.x + 2*gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z));
  do_wht_step(b0[0], b0[step]);
}
//* Compute inverse WHT of a matrix on GPU (Hadamard order), column by column, that
//* is compute the product W*M, where W is a WH matrix and M is the given
//* matrix. Result is the same matrix.
template <typename T>
void h_iwht(T *mtx,		//*< Matrix to be multiplied (in place)
	    size_t nr,		//*< Number of rows in mtx (must be a power of 2)
	    size_t nc		//*< Number of columns in mtx
	    )
{
  const size_t nr_thrds = nr>>1;
  if(nr_thrds == 0) return;

  const int log2_blk = cuda_dev->log2ThrdsBlk();
  const size_t max_thrds_blk = 1<<log2_blk;
  const size_t n_thrds_blk = nr_thrds<max_thrds_blk? nr_thrds: max_thrds_blk;

  dim3 grd0(nr_thrds/n_thrds_blk, nc);
  d_iwht0 <<< grd0, n_thrds_blk >>> (mtx);
  cudaStreamSynchronize(0);

  size_t step = 2*n_thrds_blk;
  for(; step < nr; step = step*2) {
    dim3 grd1(step/n_thrds_blk, nr_thrds/step, nc);
    d_iwht1 <<<grd1, n_thrds_blk>>> (mtx);
    cudaStreamSynchronize(0);
  }
}

//* Compute WHT of a matrix on GPU (Hadamard order), column by column, that is
//* compute the product W*M, where W is a WH matrix and M is the given matrix,
//* and then divide the result by the number of rows in W. Result is the same
//* matrix.
template <typename T>
__host__ void h_wht(T *mtx,  //*< Matrix to be multiplied (in place)
		    size_t nr, //*< Number of rows in mtx (must be a power of 2)
		    size_t nc  //*< Number of columns in mtx
		    )
{
  h_iwht(mtx, nr, nc);
  h_vec_div_scalar(T(nr), nr*nc, mtx);
}


#endif	// #ifdef __NVCC__

#endif	/* __wht_h__ */
