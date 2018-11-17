#include <stdio.h>
#include <stdlib.h>
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_dbg.h"
#include "cs_interpolate.h"

// #define  CUDA_DBG
// #define  CUDA_DBG1

__global__ void d_make_interpolate_420_1 ( int *input, int *output,
	int xdim, int ydim, int zdim, int frsize, int nxdim, int nydim,
	int nfrsize, int size
#ifdef CUDA_OBS 
	, int *cudadbgp
#endif 
	)
{
	int i, row_idx, column_idx, frame_n ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// the size is the total size on device

	while ( t_idx < size )
	{
		frame_n = t_idx / nfrsize ;

		i = t_idx % nfrsize ;
		row_idx = i / nxdim ; 

		if (!( row_idx & 1 ))
		{
			row_idx >>= 1 ;

			i %= nxdim ;
			column_idx = ( i >> 1 ) ;

			i = frame_n * frsize + row_idx * xdim + column_idx ;

			output[ t_idx ] = input [ i ] ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

// xdim/ydim/frsize are all for the new interpolated data
__global__ void d_make_interpolate_420_2 ( int *input,
	int xdim, int ydim, int frsize, int size
#ifdef CUDA_OBS 
	, int *cudadbgp
#endif 
	)
{
	int from_row_1, from_row_2, i, row_idx, column_idx, frame_n ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// the size is the total size on device

	while ( t_idx < size )
	{
		frame_n = t_idx / frsize ;

		i = t_idx % frsize ;
		row_idx = i / xdim ; 

		if ( row_idx & 1 )
		{
			from_row_1 = row_idx - 1 ;
			from_row_2 = row_idx + 1 ;

			column_idx = i % xdim ;

			if ( from_row_2 == ydim )
			{
				input[ t_idx ] = input [ frame_n * frsize +
					from_row_1 * xdim + column_idx ] ;
			} else
			{
				input[ t_idx ] = ( input [ frame_n * frsize +
					from_row_1 * xdim + column_idx ] +
					input [ frame_n * frsize +
					from_row_2 * xdim + column_idx ] ) / 2 ;
			}
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

// take care of the columns 
__global__ void d_make_interpolate_420_3 ( int *input,
	int xdim, int frsize, int size
#ifdef CUDA_OBS 
	, int *cudadbgp
#endif 
	)
{
	int from_col_1, from_col_2, i, row_idx, column_idx, frame_n ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// the size is the total size on device

	while ( t_idx < size )
	{
		frame_n = t_idx / frsize ;

		i = t_idx % frsize ;
		row_idx = i / xdim ; 
		column_idx = i % xdim ;

		if ( column_idx & 1 )
		{
			from_col_1 = column_idx - 1 ;
			from_col_2 = column_idx + 1 ;

			if ( from_col_2 != xdim )
			{
				input[ t_idx ] = ( input [ frame_n * frsize +
					row_idx * xdim + from_col_1 ] +
					input [ frame_n * frsize +
					row_idx * xdim + from_col_2 ] ) / 2 ;
			}
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

/* 
input : device addr ... also the output addr ... pls note
output : device addr
xdim : x dimension of frame 
ydim : y dimension of frame 
zdim : z dimension of frame, i.e. temporal 
scheme : INT_YUV420 currently
*/
int
h_make_interpolate ( int *d_input, int *d_output,
	int xdim, int ydim, int zdim,
	int scheme
#ifdef CUDA_OBS 
	, int *cudadbgp 
#endif 
	)
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int oframe_size, nframe_size, nn, nBlocks ;

	switch ( scheme ) {
	case INT_YUV420 :

		oframe_size = xdim * ydim ;
		nframe_size = oframe_size * 4 ; // YUV420
		nn = nframe_size * zdim  ;
		// nBlocks = ( nn + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

		h_block_adj ( nn, nThreadsPerBlock, &nBlocks ) ;

#ifdef CUDA_DBG 

		fprintf( stderr, "%s: din %p dout %p x/y/z %d %d %d sche %d\n",
			__func__,
			d_input, d_output, xdim, ydim, zdim, scheme ) ;
#endif 

		d_make_interpolate_420_1 <<< nBlocks, nThreadsPerBlock >>> ( d_input,
			d_output, xdim, ydim, zdim, oframe_size, xdim << 1, ydim << 1, 
			nframe_size, nn
#ifdef CUDA_OBS 
			, cudadbgp
#endif 
			) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
		dbg_p_d_data_i ( "make_interpolate_1", d_output, nn ) ;
#endif 

		d_make_interpolate_420_2 <<< nBlocks, nThreadsPerBlock >>> ( d_output,
			xdim << 1, ydim << 1, nframe_size, nn 
#ifdef CUDA_OBS 
			, cudadbgp
#endif 
			) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
		dbg_p_d_data_i ( "make_interpolate_2", d_output, nn ) ;
#endif 

		d_make_interpolate_420_3 <<< nBlocks, nThreadsPerBlock >>> ( d_output,
			xdim << 1, nframe_size, nn 
#ifdef CUDA_OBS 
			, cudadbgp
#endif 
			) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
		dbg_p_d_data_i ( "make_interpolate_3", d_output, nn ) ;
#endif 

		break ;
	
	default :
		return ( 0 ) ;
	}

	return ( 1 ) ;


	
}
