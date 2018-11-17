#include <stdio.h>
#include "cs_cuda.h"
#include "cs_helper.h"
#include <stdlib.h>

// #define CUDA_DBG 
// #define CUDA_DBG1 

__global__ void d_expand_frame ( int *d_input, int *d_output,
	int xdim, int ydim,
	int xadd, int yadd, int zadd, int num_of_frames,
	int o_frsize, int n_frsize, int size )
{
	int frame_idx, frame_left, o_x, o_y ;

	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < size )
	{
		frame_idx = t_idx / n_frsize ;

		if ( frame_idx >= num_of_frames )
			d_output[ t_idx ] = 0 ;
		else
		{
			frame_left = t_idx % n_frsize ;
			o_x = frame_left % ( xdim + xadd * 2 ) ;

			if ( o_x < xadd ) 
				o_x = 0 ;
			else if ( o_x >= ( xdim + xadd ))
				o_x = xdim - 1 ;
			else
				o_x -= xadd ;
			
			o_y = frame_left / ( xdim + xadd * 2 ) ;

			if ( o_y < yadd ) 
				o_y = 0 ;
			else if ( o_y >= ( yadd + ydim ))
				o_y = ydim - 1 ;
			else
				o_y -= yadd ;

			d_output[ t_idx ] = d_input[ frame_idx * o_frsize +
				o_y * xdim + o_x ] ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_expand_frame:  expand the current frames in a block
	this is supposed to be the first step after pix are copied into
	the device memory d_input

d_input	: device address of input
		should have size ( xdim * ydim * num_of_frames ) ;
d_output: device address of output
	 	should have size ( xdim + xadd * 2 ) * ( ydim + yadd * 2 ) * 
		( num_of_frames + zadd )
xdim	: frame H size
ydim	: frame V size
	
xadd	: size of H pix added on each side 
yadd	: size of V pix added on each side
zadd	: size of T pix added at the end , content value is 0
num_of_frames : with data in d_input

*/

void
h_expand_frame ( int *d_input, int *d_output, int xdim, int ydim,
	int xadd, int yadd, int zadd, int num_of_frames )
{
	int o_framesize, n_framesize, n ;
	int nThreadsPerBlock = 512;
	int nBlocks ;

#ifdef CUDA_DBG1 
	fprintf( stderr, "%s: d_input %p dout %p x/y %d %d add x/y/z %d %d %d z %d\n",
		__func__, d_input, d_output, xdim, ydim, xadd, yadd, zadd,
		num_of_frames ) ;
#endif 

	o_framesize = xdim * ydim ;
	n_framesize = ( xdim + xadd * 2 ) * ( ydim + yadd * 2 ) ;

	n = n_framesize * ( num_of_frames + zadd ) ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: old %d new %d n %d \n",
		__func__, o_framesize, n_framesize, n ) ;
#endif 

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	// nBlocks = ( n + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	d_expand_frame <<< nBlocks, nThreadsPerBlock >>> 
		( d_input, d_output,
		xdim, ydim, xadd, yadd, zadd, num_of_frames,
		o_framesize, n_framesize, n ) ;

	cudaThreadSynchronize() ;
}
