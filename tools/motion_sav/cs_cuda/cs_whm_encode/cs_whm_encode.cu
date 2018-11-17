#include <stdio.h>
#include <stdlib.h>

#include "cs_cuda.h"
#include "cs_whm_encode.h"

__global__ void d_do_a_pair_32_2_32( int *a, int size, int offset )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int f, ff ;

	if ( tid < ( size >> 1 ))
	{
		f = ( tid / offset ) * ( offset << 1 ) ;
		tid = f + tid % offset ;

		f = a[ tid ] ;
		ff = a[ tid + offset ] ;
		a[ tid ] = f + ff ;
		a[ tid + offset ] = f - ff ;
	} 
}

__global__ void d_do_a_pair_8_2_32( unsigned char *from, int size, int offset, int *to )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int f, ff ;

	if ( tid < ( size >> 1 ))
	{
		f = ( tid / offset ) * ( offset << 1 ) ;
		tid = f + tid % offset ;

		f = from[ tid ] ;
		ff = from[ tid + offset ] ;
		to[ tid ] = f + ff ;
		to[ tid + offset ] = f - ff ;
	} 
}

void 
cs_whm_measurement( char *d_from, int *d_to, int n )
{
	int first, loop, offset ;

	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;

	int nBlocks = ( n + ( nThreadsPerBlock - 1 )) / nThreadsPerBlock ;

	first = 1 ;
	loop = n ;
	loop >>= 1 ;
	offset = 1 ;
	while ( loop > 0 )
	{
#ifdef CUDA_OBS 
		fprintf( stderr, "cs_whm_measurement: f %p t %p cnt %d first %d "
			"loop %d offset %d nblk %d \n", d_from, d_to, n, first, loop,
			offset, nBlocks ) ;
#endif 

		if ( first )
		{
  			d_do_a_pair_8_2_32 <<< nBlocks, nThreadsPerBlock >>> 
				(( unsigned char * )d_from, n, offset, d_to ) ;
			first = 0 ;
		} else
  			d_do_a_pair_32_2_32 <<< nBlocks, nThreadsPerBlock >>> 
				( d_to, n, offset ) ;

		cudaThreadSynchronize() ;

		offset <<= 1 ;
		loop >>= 1 ;
	}
}
