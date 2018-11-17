#include <stdio.h>
#include <stdlib.h>
#include "cs_block.h"
#include "cs_cuda.h"
#include "cs_helper.h"

#define  CUDA_DBG

__global__ void d_make_block ( int *input, int *output,
	int xdim, int ydim, int frame_size,
	int xbdim, int ybdim, int zbdim,
	int blk_dst_size,
	int size, int do_perm,
	int overlap_x, int overlap_y,
	int x_blknum, int y_blknum, int app_x, int app_y, int weight_scheme,
	int shift
#ifdef CUDA_OBS 
	, int *cudadbgp
#endif 
	)
{
	int i, j, blk_row_idx, blk_column_idx, frame_num_in_blk,
		weight, pix_row_in_blk, pix_column_in_blk ;

	int ot_idx, t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// the size is the total size on device

	while ( t_idx < size )
	{
		ot_idx = t_idx ;

		blk_row_idx = t_idx / ( blk_dst_size * x_blknum ) ;	// which blk in row

		i = t_idx % ( blk_dst_size * x_blknum ) ;
		blk_column_idx = i / blk_dst_size ;	// which blk in column

		i = i % blk_dst_size ;	
		frame_num_in_blk = i / (( xbdim + app_x ) * ( ybdim + app_y )) ;

		if ( frame_num_in_blk < zbdim )
		{
			i %= (( xbdim + app_x ) * ( ybdim + app_y )) ;

#ifdef CUDA_OBS 
			cudadbgp[ t_idx ] = i ;
#endif 

			pix_row_in_blk = i / ( xbdim + app_x ) ;

#ifdef CUDA_OBS 
			cudadbgp[ 100 + t_idx ] = pix_row_in_blk;
#endif 
			if ( pix_row_in_blk >= app_x )
			{
				pix_row_in_blk -= app_x ;
				
				pix_column_in_blk = i % ( xbdim + app_x ) ;

#ifdef CUDA_OBS 
				cudadbgp[ 200 + t_idx ] = pix_column_in_blk ;
#endif 

				if ( pix_column_in_blk < xbdim )
				{
					// ok now we have to find the real data ...
					
					if ( weight_scheme == WEIGHT_LINEAR )
					{
						i = ybdim >> 1 ;
						weight = pix_row_in_blk % i ;
						if ( pix_row_in_blk >= i )
							weight = i - weight ;
						else
							weight++ ;

						i = xbdim >> 1 ;
						j = pix_column_in_blk % i ;
						if ( pix_column_in_blk >= i )
							j = i - j ;
						else
							j++ ;

						weight *= j ;
					} else
						weight = 1 ;

					i = frame_num_in_blk * frame_size +	
						( blk_row_idx * overlap_y + pix_row_in_blk ) * xdim +
						( blk_column_idx * overlap_x + pix_column_in_blk ) ; 

#ifdef CUDA_OBS 
					cudadbgp[ 300 + t_idx ] = i ;
#endif 

					if ( do_perm )
						t_idx++ ;
	
					output[ t_idx ] = (( input[ i ] * weight ) >> shift ) ;
				}
			}
		}

		t_idx = ot_idx + CUDA_MAX_THREADS ;
	} 
#ifdef CUDA_OBS 
	else
		cudadbgp[ 64 ] = t_idx ;
#endif 
	
}

/* 
input : device addr
output : device addr
xdim : x dimension of frame // include the expansion
ydim : y dimension of frame // include the expansion
frame_size : frame size ( ie. xdim * ydim )
xbdim : x dimension of block
ybdim : y dimension of block
zbdim : z dimension of block
blk_dst_size : block size, in elements, on device, including padding
do_perm: if permutation needed
x_overlap/y_overlap: size of advance in x/y neighboring blks
	i.e. xbdim - real_over_lap
x_blknum : how many blk in x ... consider the overlap
y_blknum : how many blk in y ... consider the overlap
app_x : after blk append app_x of 0 in x dir 
app_y : after blk append app_y of 0 in y dir 
weight_scheme : weighting scheme
shift: to avoid overflow the 'int'
*/
void
h_make_block ( int *d_input, int *d_output,
	int xdim, int ydim, int frame_size,
	int xbdim, int ybdim, int zbdim, int blk_dst_size,
	int do_perm,
	int x_overlap, int y_overlap,
	int x_blknum, int y_blknum, int app_x, int app_y, int weight_scheme,
	int shift
#ifdef CUDA_OBS 
	, int *cudadbgp 
#endif 
	)
{
	int nThreadsPerBlock = 512;
	int nn = blk_dst_size * x_blknum * y_blknum ;
	int nBlocks ; // = ( nn + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 

	fprintf( stderr, "%s: din %p dout %p x/y %d %d fsize %d blk x/y/z %d %d %d\n"
		"	blk_dst_size %d perm %d overlap x/y %d %d blks_x/y %d %d app x/y %d %d\n"
		"	weight %d nn %d shift %d\n",
		__func__,
		d_input, d_output, xdim, ydim, frame_size, xbdim, ybdim, zbdim,
		blk_dst_size, do_perm, x_overlap, y_overlap, x_blknum,
		y_blknum, app_x, app_y, weight_scheme, nn, shift ) ; 

#endif 

	h_block_adj ( nn, nThreadsPerBlock, &nBlocks ) ;

	d_make_block <<< nBlocks, nThreadsPerBlock >>> ( d_input, d_output,
		xdim, ydim, frame_size, xbdim, ybdim, zbdim, 
		blk_dst_size, nn, do_perm,
		x_overlap, y_overlap, 
		x_blknum, y_blknum, app_x, app_y, weight_scheme,
		shift
#ifdef CUDA_OBS 
		, cudadbgp
#endif 
		) ;

	cudaThreadSynchronize() ;
}
