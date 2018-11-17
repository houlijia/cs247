#include <stdio.h>
#include <stdlib.h>

#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_whm_encode_b.h"

template<typename T>
__global__ void d_divide_by( T *a, int size, int blk_size )
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int otid ;

  otid = tid ;
  while ( tid < size )
    {
      a[ tid ] = a[ tid ] / blk_size ;

      otid += CUDA_MAX_THREADS ;
      tid = otid ;
    } 
}

template<typename T>
__global__ void 
d_do_a_pair_32_2_32_b( T *a, int size, int offset, int blk_size
#ifdef CUDA_DBG 
		       , int *p
#endif 
		       )
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int otid, i ;
  T *oa, f, ff ;

  oa = a ;
  otid = tid ;
  while ( tid < size )
    {
      a = oa ;

      i = ( tid / blk_size ) * blk_size ;
      tid -= i ;
		
      if ( tid < ( blk_size >> 1 ))
	{
	  a += i ;

	  f = ( tid / offset ) * ( offset << 1 ) ;
	  tid = f + tid % offset ;

	  f = a[ tid ] ;
	  ff = a[ tid + offset ] ;
	  a[ tid ] = f + ff ;
	  a[ tid + offset ] = f - ff ;
	}

      otid += CUDA_MAX_THREADS ;
      tid = otid ;
		
    } 
}

template<>
__global__ void d_do_a_pair_32_2_32_b( int *a, int size, int offset,
				       int blk_size
#ifdef CUDA_DBG 
				       , int *p
#endif 
				       )
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int *oa, otid, i, f, ff ;

  oa = a ;
  otid = tid ;
  while ( tid < size )
    {
      a = oa ;

      i = ( tid / blk_size ) * blk_size ;
      tid -= i ;
		
      if ( tid < ( blk_size >> 1 ))
	{
	  a += i ;

	  f = ( tid / offset ) * ( offset << 1 ) ;
	  tid = f + tid % offset ;

	  f = a[ tid ] ;
	  ff = a[ tid + offset ] ;
	  a[ tid ] = f + ff ;
	  a[ tid + offset ] = f - ff ;
	}

      otid += CUDA_MAX_THREADS ;
      tid = otid ;
		
    } 
}

template<>
__global__ void 
d_do_a_pair_32_2_32_b( float *a, int size, int offset, int blk_size
#ifdef CUDA_DBG 
			      , int *p
#endif 
			      )
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int otid, i ;
  float *oa, f, ff ;

  oa = a ;
  otid = tid ;
  while ( tid < size )
    {
      a = oa ;

      i = ( tid / blk_size ) * blk_size ;
      tid -= i ;
		
      if ( tid < ( blk_size >> 1 ))
	{
	  a += i ;

	  f = ( tid / offset ) * ( offset << 1 ) ;
	  tid = f + tid % offset ;

	  f = a[ tid ] ;
	  ff = a[ tid + offset ] ;
	  a[ tid ] = f + ff ;
	  a[ tid + offset ] = f - ff ;
	}

      otid += CUDA_MAX_THREADS ;
      tid = otid ;
		
    } 
}

template<typename T>
void 
cs_whm_measurement_b( T *d_from, int n, int blk_size )
{
  int loop, offset ;

#ifdef CUDA_DBG 
  int *d_dbgp ;
#endif 

  int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

  int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

  h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

  loop = blk_size ;
  loop >>= 1 ;
  offset = 1 ;

#ifdef CUDA_OBS 
  fprintf( stderr, "%s: f %p n %d blk %d loop %d offset %d\n",
	   __func__, d_from, n, blk_size, loop, offset ) ;
#endif 

#ifdef CUDA_OBS 
  d_dbgp = dbg_d_malloc_i ( 512 ) ;
  if ( !d_dbgp )
    {
      fprintf( stderr, "%s: dbg_d_malloc failed \n", __func__ ) ;
      exit( 1 ) ;
    }
  clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

  d_do_a_pair_32_2_32_b<T> <<< nBlocks, nThreadsPerBlock >>> 
    ( d_from, n, offset, blk_size
#ifdef CUDA_DBG 
      , d_dbgp 
#endif 
      ) ;

  cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
  dbg_p_d_data_i ("dbg:after first loop", d_dbgp, 512 ) ;
  clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

  offset <<= 1 ;
  loop >>= 1 ;

  while ( loop > 0 )
    {
#ifdef CUDA_OBS 
      if ( blk_size < offset )
	{
	  fprintf( stderr, "%s: f %p t %p cnt %d blksize %d "
		   "loop %d offset %d nblk %d \n", __func__, d_from, d_from, n, blk_size, loop,
		   offset, nBlocks ) ;	
	}
#endif 

      d_do_a_pair_32_2_32_b<T> <<< nBlocks, nThreadsPerBlock >>> 
	( d_from, n, offset, blk_size
#ifdef CUDA_DBG 
	  , d_dbgp
#endif 		 
	  ) ;

      cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
      dbg_p_d_data_i ("dbg:in loop", d_dbgp, 512 ) ;
      clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

      offset <<= 1 ;
      loop >>= 1 ;
    }
}

template <> void 
cs_whm_measurement_b( int *d_from, int n, int blk_size )
{
  int loop, offset ;

#ifdef CUDA_DBG 
  int *d_dbgp ;
#endif 

  int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

  int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

  h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

  loop = blk_size ;
  loop >>= 1 ;
  offset = 1 ;

#ifdef CUDA_OBS 
  fprintf( stderr, "%s: f %p n %d blk %d loop %d offset %d\n",
	   __func__, d_from, n, blk_size, loop, offset ) ;
#endif 

#ifdef CUDA_OBS 
  d_dbgp = dbg_d_malloc_i ( 512 ) ;
  if ( !d_dbgp )
    {
      fprintf( stderr, "%s: dbg_d_malloc failed \n", __func__ ) ;
      exit( 1 ) ;
    }
  clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

  d_do_a_pair_32_2_32_b <<< nBlocks, nThreadsPerBlock >>> 
    ( d_from, n, offset, blk_size
#ifdef CUDA_DBG 
      , d_dbgp
#endif 		 
      ) ;

  cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
  dbg_p_d_data_i ("dbg:in loop", d_dbgp, 512 ) ;
  clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

  offset <<= 1 ;
  loop >>= 1 ;
}

// float begin

template <> void 
cs_whm_measurement_b( float *d_from, int n, int blk_size )
{
	int loop, offset ;

#ifdef CUDA_DBG 
	int *d_dbgp ;
#endif 

	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

	int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	loop = blk_size ;
	loop >>= 1 ;
	offset = 1 ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: f %p n %d blk %d loop %d offset %d\n",
		__func__, d_from, n, blk_size, loop, offset ) ;
#endif 

#ifdef CUDA_OBS 
	d_dbgp = dbg_d_malloc_i ( 512 ) ;
	if ( !d_dbgp )
	{
		fprintf( stderr, "%s: dbg_d_malloc failed \n", __func__ ) ;
		exit( 1 ) ;
	}
	clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

	d_do_a_pair_32_2_32_b <<< nBlocks, nThreadsPerBlock >>> 
		( d_from, n, offset, blk_size
#ifdef CUDA_DBG 
		, d_dbgp 
#endif 
		) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i ("dbg:after first loop", d_dbgp, 512 ) ;
	clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

	offset <<= 1 ;
	loop >>= 1 ;

	while ( loop > 0 )
	{
#ifdef CUDA_OBS 
		if ( blk_size < offset )
		{
			fprintf( stderr, "%s: f %p t %p cnt %d blksize %d "
				"loop %d offset %d nblk %d \n", __func__, d_from, d_from, n, blk_size, loop,
				offset, nBlocks ) ;	
		}
#endif 

		d_do_a_pair_32_2_32_b <<< nBlocks, nThreadsPerBlock >>> 
			( d_from, n, offset, blk_size
#ifdef CUDA_DBG 
			, d_dbgp
#endif 		 
			) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
		dbg_p_d_data_i ("dbg:in loop", d_dbgp, 512 ) ;
		clear_device_mem_i ( d_dbgp, 512 ) ;
#endif 

		offset <<= 1 ;
		loop >>= 1 ;
	}
}

template<typename T>
void 
cs_iwhm_measurement_b( T *d_from, int n, int blk_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

	int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	cs_whm_measurement_b<T>( d_from, n, blk_size ) ;

	d_divide_by<T> <<< nBlocks, nThreadsPerBlock >>> 
		( d_from, n, blk_size ) ;

	cudaThreadSynchronize() ;
}

template <>void 
cs_iwhm_measurement_b( float *d_from, int n, int blk_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

	int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	cs_whm_measurement_b( d_from, n, blk_size ) ;

	d_divide_by <<< nBlocks, nThreadsPerBlock >>> 
		( d_from, n, blk_size ) ;

	cudaThreadSynchronize() ;
}

// float end

template <> void 
cs_iwhm_measurement_b( int *d_from, int n, int blk_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

	int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	cs_whm_measurement_b( d_from, n, blk_size ) ;

	d_divide_by <<< nBlocks, nThreadsPerBlock >>> 
		( d_from, n, blk_size ) ;

	cudaThreadSynchronize() ;
}

// Explicit template instantiations

template __global__ void 
d_do_a_pair_32_2_32_b<>( int *a, int size, int offset,
			 int blk_size
#ifdef CUDA_DBG 
			 , int *p
#endif 
			 );
template __global__ void 
d_do_a_pair_32_2_32_b<>( float *a, int size, int offset, int blk_size
#ifdef CUDA_DBG 
			 , int *p
#endif 
			 );


template void 
cs_whm_measurement_b( int *d_from, int n, int blk_size );

template void 
cs_whm_measurement_b( float *d_from, int n, int blk_size );
