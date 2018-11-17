#include <stdio.h>
#include <stdlib.h>

#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_whm_encode_b.h"

#undef CUDA_DBG

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

__global__ void d_do_a_pair_8_2_32_b( unsigned char *from, int size,
	int offset, int *to, int blk_size
#ifdef CUDA_DBG 
	, int *p 
#endif 
	)
{
#ifdef CUDA_DBG 
	int otid ;
#endif 
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int i, f, ff ;

	if ( tid < size )
	{
		i = ( tid / blk_size ) * blk_size ;

#ifdef CUDA_DBG 
		otid = tid ;
#endif 
		tid -= i ;
		
		if ( tid < ( blk_size >> 1 ))
		{
			f = ( tid / offset ) * ( offset << 1 ) ;
			tid = f + tid % offset ;

			from += i ;
			to += i ;
#ifdef CUDA_DBG 
			p[otid] = tid ;
#endif 
			f = from[ tid ] ;
			ff = from[ tid + offset ] ;
			to[ tid ] = f + ff ;
			to[ tid + offset ] = f - ff ;
		}
	} 
}

void 
cs_whm_measurement_b( int *d_from, int n, int blk_size )
{
	int loop, offset ;

#ifdef CUDA_DBG 
	int *d_dbgp ;
#endif 

	int nThreadsPerBlock = 512;

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
