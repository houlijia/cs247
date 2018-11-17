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
#include "cs_header.h"
#include "cs_helper.h"
#include "cs_analysis.h"
#include "cs_edge_detect_v2.h"
#include "cs_copy_box.h"

// #define CUDA_DBG
// #define CUDA_DBG1

// the blk here refers to the L-selection block ... i.e. cube
// bxyz_size: the L-selected size // inner block size ( which is bigger than
//		the edge/corner block size, but we use the inner block in the computation
// exy_size: edge_rectangle - 1
// 			: ( edge_x * 2 + 1 ) * ( edge_y * 2 + 1 ) - 1
// tbl_size: overall size for inner block
//		should be ( xblock * yblock * zblock * nblock_in_x * nblock_in_y )
// note : both fdp and tdp will point to the same size block size
//		the only diff is that the fdp will have all other values after L-selection
//		and the tdp will have the "edged" block surrounded by the value 0
//		more entries will have the value 0 if edge/corner blocks 

template<typename T>
__global__ void d_do_edge_detection_v2 ( T *fdp, T *tdp, int tbl_size, 
	struct cube *xyzp, int ex, int ey, int bxyz_size, int exy_size,
	int blk_in_x, int blk_in_y )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int cx, cy, cz, xy_size, xyz_idx, x, y, frame, block, i, j ;
	float d ;
	T mea, sum, *ofdp, *fp ; 

#ifdef CUDA_OBS 
	if ( t_idx == 0 )
	{
		fp = tdp + 168 ;
		*fp++ = tbl_size ;
		*fp++ = ex ;
		*fp++ = ey ;
		*fp++ = bxyz_size ;
		*fp++ = exy_size ;
		*fp++ = blk_in_x ;
		*fp++ = blk_in_y ;
		*fp++ = xyzp[0].x ;
		*fp++ = xyzp[0].y ;
		*fp++ = xyzp[0].z ;
		*fp++ = xyzp[1].x ;
		*fp++ = xyzp[1].y ;
		*fp++ = xyzp[1].z ;
		*fp++ = xyzp[2].x ;
		*fp++ = xyzp[2].y ;
		*fp++ = xyzp[2].z ;
	}
#endif 

	ofdp = fdp ;
	while ( t_idx < tbl_size )
	{
		fdp = ofdp ;

		block = t_idx / bxyz_size ; // which block that this measurement sits 

#ifdef CUDA_OBS 
		fp = tdp + 168 * 2 ;

		*fp++ = block ;
#endif 

		j = block / blk_in_x ;	// 0..(blk_in_y-1)
		i = block % blk_in_x ;	// 0..(blk_in_x-1)

		if (( i == 0 )|| ( i == ( blk_in_x - 1 )))
		{
			if (( j == 0 ) || ( j == ( blk_in_y - 1 ))) 
				xyz_idx = 2 ;
			else
				xyz_idx = 1 ;
		} else
		{
			if (( j == 0 ) || ( j == ( blk_in_y - 1 ))) 
				xyz_idx = 1 ;
			else
				xyz_idx = 0 ;
		}

		cx = xyzp[ xyz_idx ].x ;
		cy = xyzp[ xyz_idx ].y ;
		cz = xyzp[ xyz_idx ].z ;

#ifdef CUDA_OBS 
		*fp++ = i ;
		*fp++ = j ;
		*fp++ = cx ;
		*fp++ = cy ;
#endif 

		i = t_idx % bxyz_size ;	// the offset of this measurement 
							// in this block (inner/edge/corner)

		xy_size = cx * cy ;

		frame = i / xy_size ;
		i %= xy_size ;	// offset of mea in this frame

		y = i / cx ;
		x = i % cx ;

		xy_size = cx * cy ;

#ifdef CUDA_OBS 
		*fp++ = i ;
		*fp++ = x ;
		*fp++ = y ;
		*fp++ = cx ;
		*fp++ = cy ;
		*fp++ = xy_size ;
#endif 
		

		if (( frame < cz ) && ( y >= ey ) && 
			( x >= ex ) && (( cy - y ) > ey ) && (( cx -x ) > ex )) 
		{
			mea = fdp[ t_idx ] ;

#ifdef CUDA_OBS 
			*fp++ = mea ;
#endif 

			fdp += block * bxyz_size + frame * xy_size ;
				// the offset of this frame in the blk

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

			d = ((( float ) sum ) / (( float ) exy_size )) ;
		
			// round up

			tdp [ t_idx ] = ( T )d - mea ;

			// template FIX ... tdp [ t_idx ] = (( int )( d + 0.5 )) - mea ;
			// tdp [ t_idx ] = xyz_idx ;

		} else
			tdp [ t_idx ] = 0 ;

		t_idx += CUDA_MAX_THREADS ;
	}		
}

// edge_x/y are from the center of the edge box on each side
// fromp will have the final data ... since we do the copy box
// tbl size is cube_x * cube_y * cube_z * nblok_in_x * nblock_in_y ( of the inner cube )

template<typename T>
int
h_do_edge_detection_v2 ( T *fromp, T *top, int tbl_size,
	struct cube *d_xyzp, int edge_x, int edge_y, int blk_in_x,
	int blk_in_y, struct cube *cubep )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	int edge_xy = ( edge_x * 2 + 1 ) * ( edge_y * 2 + 1 ) - 1 ;
	int i, bxyz_size = cubep[0].x * cubep[0].y * cubep[0].z ; // inner

#ifdef CUDA_DBG1 
	fprintf(stderr, "%s: f %p t %p xyzp %p tblsize %d edge %d %d\n",
		__func__, fromp, top, d_xyzp, tbl_size, edge_x, edge_y ) ;
	fprintf(stderr, "	: exy %d bxyz %d blk_in_x/y %d %d cubep %p\n",
		edge_xy, bxyz_size, blk_in_x, blk_in_y, cubep ) ;
#endif 

	if ( tbl_size % bxyz_size )
	{
		fprintf(stderr, "%s: error size %d cube %d \n", __func__,
			tbl_size, bxyz_size ) ;
		return ( 0 ) ;
	}

	i = tbl_size / bxyz_size ;

	if ( i != ( blk_in_x * blk_in_y ))
	{
		fprintf(stderr, "%s: #_of_block %d blk_in_x/y %d %d\n",
			__func__, i, blk_in_x, blk_in_y ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_OBS 
	// ck in the allocate_d_mem already ...
	f ((( cube_x - ( edge_x * 2 + 1 )) < 0 ) ||
		(( cube_y - ( edge_y * 2 + 1 )) < 0 ))
	{
		fprintf(stderr, "%s: error cube %d %d edge %d %d\n",
			__func__, cube_x, cube_y, edge_x, edge_y ) ;
		return ( 0 ) ;
	}
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_edge_detection_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top, tbl_size, d_xyzp, edge_x, edge_y, 
		bxyz_size, edge_xy, blk_in_x, blk_in_y ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("edge_detect", top, tbl_size ) ; 
#endif 
	
	// QQQ need to copy differently 

#ifdef CUDA_OBS 

	if ( !h_do_copy_box ( top, fromp, tbl_size, cube_x,
		cube_y, edge_x, edge_y ))
	{
		return ( 0 ) ;
	}
#endif 

	return ( 1 ) ;
}

template int
h_do_edge_detection_v2<int> ( int *fromp, int *top, int tbl_size,
	struct cube *d_xyzp, int edge_x, int edge_y, int blk_in_x,
	int blk_in_y, struct cube *cubep ) ;

template int
h_do_edge_detection_v2<float> ( float *fromp, float *top, int tbl_size,
	struct cube *d_xyzp, int edge_x, int edge_y, int blk_in_x,
	int blk_in_y, struct cube *cubep ) ;
