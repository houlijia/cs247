#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_cuda.h"
#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_edge_detect.h"
#include "cs_copy_box.h"

// #define CUDA_DBG
// #define CUDA_DBG1

__global__ void d_do_edge_detection ( int *fdp, int *tdp, int tbl_size, 
	int cx, int cy, int ex, int ey, int xy_size, int exy_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *ofdp, mea, *fp, x, y, block, sum, i, j ;
	double d ;

	ofdp = fdp ;
	while ( t_idx < tbl_size )
	{
		fdp = ofdp ;
		i = t_idx % xy_size ;
		y = i / cx ;
		x = i % cx ;

		if (( y >= ey ) && ( x >= ex ) && (( cy - y ) > ey ) && (( cx -x ) > ex )) 
		{
			mea = fdp[ t_idx ] ;

			block = t_idx / xy_size ;
			fdp += block * xy_size ;

			sum = 0 ;
			for ( j = -ey ; j <= ey ; j++ )
			{
				fp = fdp + ( y + j ) * cx + ( x - ex ) ;

				for ( i = -ex ; i <= ex ; i++ )
				{
					sum += *fp++ ;
				} 
			}

			sum -= mea ;

			// exy_size take out the one in the center already ...

			d = ((( double ) sum ) / (( double ) exy_size )) ;
		
			// round up

			tdp [ t_idx ] = (( int )( d + 0.5 )) - mea ;

		} else
			tdp [ t_idx ] = 0 ;

		t_idx += CUDA_MAX_THREADS ;
	}		
}

// edge_x/y are from the center of the edge box on each side
// fromp will have the final data ... since we do the copy box

int
h_do_edge_detection ( int *fromp, int *top, int tbl_size, int cube_x,
	int cube_y, int edge_x, int edge_y )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	int cube_xy = cube_y * cube_x ;
	int edge_xy = ( edge_x * 2 + 1 ) * ( edge_y * 2 + 1 ) - 1 ;

#ifdef CUDA_DBG1 
	fprintf(stderr, "%s: f %p t %p tblsize %d cube %d %d edge %d %d\n",
		__func__, fromp, top, tbl_size, cube_x, cube_y, edge_x, edge_y ) ;
#endif 

	if ( tbl_size % cube_xy )
	{
		fprintf(stderr, "%s: error size %d cube %d \n", __func__,
			tbl_size, cube_xy ) ;
		return ( 0 ) ;
	}

	if ((( cube_x - ( edge_x * 2 + 1 )) < 0 ) ||
		(( cube_y - ( edge_y * 2 + 1 )) < 0 ))
	{
		fprintf(stderr, "%s: error cube %d %d edge %d %d\n",
			__func__, cube_x, cube_y, edge_x, edge_y ) ;
		return ( 0 ) ;
	}

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_edge_detection <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top, tbl_size, cube_x, cube_y, edge_x, edge_y,
		cube_xy, edge_xy ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("edge_detect", top, tbl_size ) ; 
#endif 
	
	if ( !h_do_copy_box ( top, fromp, tbl_size, cube_x,
		cube_y, edge_x, edge_y ))
	{
		return ( 0 ) ;
	}

	return ( 1 ) ;
}
