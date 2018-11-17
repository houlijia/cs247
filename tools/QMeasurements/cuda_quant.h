#ifndef __CUDA_QUANT_HDR__
#define __CUDA_QUANT_HDR__
/** File
This file contains functions for performing the operations of linear
quantization on the GPU. The functions are written as templates in  order to
allow both single and double precision implementations.

In the templates FLT (float or double)is the class of the measurments, QV is
the class of quantized labels (prior to saturation and LBL is the class of the
labels after saturation. 
 */

#include <stddef.h>
#include <math.h>
#include <stdio.h>

#include "CudaDevInfo.h"

#include "cuda_flt_funcs.h"
#include "fast_heap.h"

template<class FLT, class QV, class LBL>
__global__ void
d_quant(size_t n_no_clip, //!< No. of measurements which are not clipped.
	size_t n_clip, //!< No. of measurements which are to be clipped.
	const FLT *vec_no_clip, //!< An array of \c n_no_clip entries to quantize
	const FLT *vec_clip, //!< An array of \c n_clip entries to quantize
	FLT intvl,      //!< The quantization interval
	FLT offset,     //!< Quantizaton offset
	LBL sat_lvl,	//!< Saturation level. If 0, no saturation
	QV *q_no_clip,	 //!< A vector of length n_no_clip.
	LBL *q_clip, //!< return: quantization labels (\c n_vec elements)
	QV *save   //!< return: saturated values (\c n_save elements)
	)
{
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_vec = n_no_clip + n_clip;
  if(j>=n_vec) 
    return;
  
  if(j<n_no_clip)
    q_no_clip[j] = (QV) cuda_flt_ceil((vec_no_clip[j]+offset)/intvl - FLT(0.5));
  else {
    j = j-n_no_clip;
    QV qval =  (QV) cuda_flt_ceil((vec_clip[j]+offset)/intvl - FLT(0.5));
   
    if(qval < 1 || qval >= (QV)sat_lvl) {
      q_clip[j] = sat_lvl;
      
      if(save != NULL)
	save[j] = qval;
    }
    else
      q_clip[j] = (LBL) qval;
  }
}

/* This kernel function is not optimized it runs sequentially. If we start using it
   it should be optimized
*/
template<class QV, class LBL>
__global__ void
d_quant_organize_save(LBL sat_lvl,
		      size_t n_save,
		      const LBL *q_clip,
		      QV *save,
		      size_t *save_cnt
		      )
{
  size_t j;
  size_t cnt = 0;

  for(j=0; j<n_save; j++) {
    if(q_clip[j] != sat_lvl)
      continue;    
    else if(save[j] < 1)
      save[cnt++] = save[j];
    else
      save[cnt++] = save[j] - (sat_lvl-1);
  }
  *save_cnt = cnt;
}

/** Quantization function. For each entry x the quntized value q is 
     <tt>    q = ceil((x+offset)/intvl) </tt>
    \c q is non-saturated if \c 0<q<sat_lvl and otherwise it is saturated. The
     coded labels output consists of three parts:
     \li \c q_no_clip - a vector containing the first \c n_no_clip quantized
     measurements, as is.
     \li \c q_clip - a vector containing the remaining \c n_vec-n_no_clip
     quantized measurements, where the value of each quantized measurements is
     set to \c sat_lvl.
     \li \c save (if not NULL) contains the original value of each of the
     saturated measurements in \c q_clip.
     \par EXAMPLE:
     Suppose the input vector was quantized in to -2,3,5,1,2,7, \c sat_lvl=4 and
     \c n_no_clip=2. Then \c q_no_clip will be -2,3, \c q_clip will be 4,1,2,4
     and
     \li if \c save is NULL then \c save_cnt is not set.
     \li if \c save is non-NULL save will contain the values 5,7 and \c save_cnt=2.

    \retrun the number of saturated values
 */
template<class FLT, class QV, class LBL>
void h_quant(size_t n_no_clip, //!< No. of measurements which are not clipped.
	     size_t n_clip, //!< No. of measurements which are to be clipped.
	     const FLT *vec_no_clip, //!< An array of \c n_no_clip entries to quantize
	     const FLT *vec_clip, //!< An array of \c n_clip entries to quantize
	     FLT intvl,      //!< The quantization interval
	     FLT offset,     //!< Quantizaton offset
	     LBL sat_lvl,	//!< Saturation level. If 0, no saturation
	     QV *q_no_clip,	 //!< A vector of length n_no_clip.
	     LBL *q_clip, //!< return: quantization labels (\c n_vec-n_no_clip elements)
	     QV *save,   //!< return: saturated values (\c n_n_vec-n_no_clip elements)
	     size_t *save_cnt	//!< return: No. of saved values (undefined if save==NULL)
	     )
{
  const size_t max_thr_blk = (size_t) cuda_dev->getProp().maxThreadsPerBlock;

  const size_t n_vec = n_no_clip + n_clip;
  int n_blks = (n_vec + max_thr_blk -1)/max_thr_blk; 
  int n_thrds_per_blk = (n_vec < max_thr_blk)? n_vec: max_thr_blk;

  d_quant <<< n_blks, n_thrds_per_blk >>>
    (n_no_clip, n_clip, vec_no_clip, vec_clip, intvl, offset, sat_lvl, q_no_clip, q_clip, save);

  cudaStreamSynchronize(0);

  if(save != NULL) {
    GenericHeapElement &ghe = d_fast_heap->get(sizeof(size_t));
    size_t *d_save_cnt = static_cast<size_t*>(*ghe);
    
    d_quant_organize_save <<<1,1>>> (sat_lvl, n_clip, q_clip, save, d_save_cnt);

    cudaStreamSynchronize(0);

    gpuErrChk(cudaMemcpy(save_cnt, d_save_cnt, sizeof(size_t), cudaMemcpyDeviceToHost),
	      "h_quant:memcpy", "copy to host failed");

    ghe.discard();
  }
}

#endif	/* __CUDA_QUANT_HDR__ */
