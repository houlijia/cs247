#ifndef __CUDA_FLT_FUNCS_HDR__
#define __CUDA_FLT_FUNCS_HDR__

/** \file
This file contains template of mathematical functions which resolve to the
appropriate single or doulbe precision mathematical functions in CUDA.
*/

template <class T>
__device__ T cuda_flt_ceil(T x) { return ceil(x);}
template <> __device__ float cuda_flt_ceil(float x) { return ceilf(x); }

template <class T>
__device__ T cuda_flt_floor(T x) { return floor(x);}
template <> __device__ float cuda_flt_floor(float x) { return floorf(x); }

#endif	/* __CUDA_FLT_FUNCS_HDR__ */
