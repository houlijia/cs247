#include <stdio.h>
#include <stdlib.h>

#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_whm_transp.h"

// SYNTAX free ... but not yet tested ... since it is the same as cs_iwhm_measurement_b()

#undef CUDA_DBG

__global__ void 
d_do_whm_transp( float *a, int offset, int blk_size )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int otid, i, f, ff ;
	float *oa ;

	oa = a ;
	otid = tid ;
	while ( tid < blk_size )
	{
		a = oa ;

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

/* 
	h_do_whm_transp:
	d_from: the vector address on device
	n: size of vector ... has to be a power of 2
	note: the data will be changed ... i.e. old data is gone when done
*/

int 
h_do_whm_transp( float *d_from, int blk_size )
{
	int loop, offset ;

#ifdef CUDA_DBG 
	float *d_dbgp ;
#endif 

	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

	int nBlocks ; //= ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

	loop = max_log2( blk_size ) ;

	if ( loop != blk_size )
	{
#ifdef CUDA_DBG 
		fprintf( stderr, "%s: blk_size %d log2 %d\n",
			__func__, blk_size, loop ) ;
#endif 
		return ( 0 ) ;
	}

	h_block_adj ( blk_size, nThreadsPerBlock, &nBlocks ) ;

	loop = blk_size ;
	loop >>= 1 ;
	offset = 1 ;

	d_do_whm_transp <<< nBlocks, nThreadsPerBlock >>> ( d_from, offset, blk_size ) ;

	cudaThreadSynchronize() ;

	offset <<= 1 ;
	loop >>= 1 ;

	while ( loop > 0 )
	{
		d_do_whm_transp <<< nBlocks, nThreadsPerBlock >>> 
			( d_from, offset, blk_size ) ;

		cudaThreadSynchronize() ;

		offset <<= 1 ;
		loop >>= 1 ;
	}
	return ( 1 ) ;
}
