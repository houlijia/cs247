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
#include "cs_copy_box.h"

// #define CUDA_DBG
// #define CUDA_DBG1

// ex, ey, and exy_size are the embedded dimensions/size
// ex/ey are from the point to the edge of the cx/cy

__global__ void d_do_copy_box ( int *fdp, int *tdp, int tbl_size, 
	int cx, int cy, int ex, int ey, int xy_size, int exy_size )
{
	int *otdp, f_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int block, i, x, y ;

	otdp = tdp ;
	while ( f_idx < tbl_size )
	{
		tdp = otdp ;

		block = f_idx / xy_size ;

		tdp += block * exy_size ;
		
		i = f_idx % xy_size ;
		y = i / cx ;
		x = i % cx ;

		if (( y >= ey ) && ( x >= ex ) && (( cy - y ) > ey ) && (( cx -x ) > ex )) 
		{
			i = ( y - ey ) * ( cx - 2 * ex ) + ( x - ex ) ;
			
			tdp [ i ] = fdp [ f_idx ] ;
		} 

		f_idx += CUDA_MAX_THREADS ;
	}		
}

// edge_x/y are the distance from the cube_x/y to the embedded box ( defined
// by edge_x/y )
int
h_do_copy_box ( int *fromp, int *top, int tbl_size, int cube_x,
	int cube_y, int edge_x, int edge_y )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	int cube_xy = cube_y * cube_x ;
	int edge_xy = (( cube_x - ( edge_x * 2 )) *
		(( cube_y - ( edge_y * 2 )))) ;

#ifdef CUDA_DBG1 
	fprintf(stderr, "%s: f %p t %p tblsize %d cube %d %d edge %d %d csize %d"
		" esize %d\n",
		__func__, fromp, top, tbl_size, cube_x, cube_y, edge_x, edge_y,
		cube_xy, edge_xy ) ;
#endif 

	if ( tbl_size % cube_xy )
	{
		fprintf(stderr, "%s: error size %d cube %d \n", __func__,
			tbl_size, cube_xy ) ;
		return ( 0 ) ;
	}

	if ((( cube_x - ( edge_x * 2 )) < 0 ) ||
		(( cube_y - ( edge_y * 2 )) < 0 ))
	{
		fprintf(stderr, "%s: error cube %d %d edge %d %d\n",
			__func__, cube_x, cube_y, edge_x, edge_y ) ;
		return ( 0 ) ;
	}

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_copy_box <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top, tbl_size, cube_x, cube_y, edge_x, edge_y,
		cube_xy, edge_xy ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("copy_box", top, ( tbl_size / cube_xy ) * edge_xy ) ; 
#endif 
	return ( 1 ) ;
}


__global__ void d_do_copy_vec ( int *fdp, int *tdp, int tbl_size, 
	int from_size, int to_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int k ;

	while ( t_idx < tbl_size )
	{
		k = ( t_idx / to_size ) * from_size + ( t_idx % to_size ) ;

		tdp[ t_idx ] = fdp[ k ] ;

		t_idx += CUDA_MAX_THREADS ;
	}		
}

// total_size is the size of the copy ... in element
// copy the first from_size elements from fromp to top for every frame
int
h_do_copy_vec ( int *fromp, int *top, int total_size, int from_size,
	int to_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( total_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1
	fprintf(stderr, "%s: f %p t %p total %d from %d to %d \n",
		__func__, fromp, top, total_size, from_size, to_size ) ; 
#endif

	if (( total_size % to_size ) || ( to_size > from_size ))
	{
		fprintf( stderr, "%s: size %d %d %d\n", total_size, from_size, to_size ) ;
		return ( 0 ) ;
	} 

	h_block_adj ( total_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_copy_vec <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top, total_size, from_size, to_size ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("copy_vec", top, total_size ) ; 
#endif 
	return ( 1 ) ;
}

// ex, ey, and exy_size are the embedded dimensions/size
// ex/ey are from the point to the edge of the cx/cy
// obxyz_size is the old inner block size
// nbxyz_size is the new inner block size 
// tbl_size is the old size

__global__ void d_do_copy_box_v2 ( int *fdp, int *tdp, int tbl_size, 
	int ex, int ey, int obxyz_size, int nbxyz_size, struct cs_xyz *d_xyzp,
	int blk_in_x, int blk_in_y )
{
	int f_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *otdp, cx, cy, cz, xy_size, exy_size, blk_idx, frame, block, i, j, x, y ;

	otdp = tdp ;
	while ( f_idx < tbl_size )
	{
		tdp = otdp ;

		block = f_idx / obxyz_size ; // which block

		i = block % blk_in_x ; 	// 0..blk_in_x-1
		j = block / blk_in_x ;	// 0..blk_in_y-1

		if (( i == 0 ) || ( i == ( blk_in_x - 1 ))) 
		{
			if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
				blk_idx = 2 ;
			else
				blk_idx = 1 ;
		} else
		{
			if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
				blk_idx = 1 ;
			else
				blk_idx = 0 ;
		}

		i = f_idx % obxyz_size ;	// mea offset in the block

		cx = d_xyzp[ blk_idx ].x ;
		cy = d_xyzp[ blk_idx ].y ;
		cz = d_xyzp[ blk_idx ].z ;

		xy_size = cx * cy ;
		exy_size = ( cx - ex * 2 ) * ( cy - ey * 2 ) ;

		frame = i / xy_size ;
		
		i = i % xy_size ;
		y = i / cx ;
		x = i % cx ;

		if (( frame < cz ) && ( y >= ey ) && ( x >= ex ) && 
			(( cy - y ) > ey ) && (( cx - x ) > ex )) 
		{
			tdp += block * nbxyz_size + frame * exy_size ;
			i = ( y - ey ) * ( cx - 2 * ex ) + ( x - ex ) ;
			
			tdp [ i ] = fdp [ f_idx ] ;
		} 

		f_idx += CUDA_MAX_THREADS ;

	}		
}

int
h_do_copy_box_v2 ( int *fromp, int *top, int tbl_size, 
	int edge_x, int edge_y, int blk_in_x, int blk_in_y, struct cs_xyz *d_cp, 
	struct cube *cp )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	int obxyz_size = cp[0].x * cp[0].y * cp[0].z ;
	int nbxyz_size = (( cp[0].x - ( edge_x * 2 )) *
		(( cp[0].y - ( edge_y * 2 ))) * cp[0].z ) ;
#ifdef CUDA_DBG 
	int i ;
#endif 
	struct cube temp_cube[ CUBE_INFO_CNT ] ;

	memcpy ( &temp_cube, cp, sizeof ( *cp )) ;

#ifdef CUDA_DBG1 
	fprintf(stderr, "%s: f %p t %p tblsize %d edge %d %d blk %d %d cubep %p\n",
		__func__, fromp, top, tbl_size, edge_x, edge_y, blk_in_x,
		blk_in_y, cp ) ;
	fprintf(stderr, " 	nsize %d osize %d\n", nbxyz_size, obxyz_size ) ;
#endif 

	if ( tbl_size % obxyz_size )
	{
		fprintf(stderr, "%s: error size %d cube %d \n", __func__,
			tbl_size, obxyz_size ) ;
		return ( 0 ) ;
	}

	if ((( cp[2].x - ( edge_x * 2 )) < 0 ) ||
		(( cp[2].y - ( edge_y * 2 )) < 0 ))
	{
		fprintf(stderr, "%s: error cube %d %d edge %d %d\n",
			__func__, cp[2].x, cp[2].y, edge_x, edge_y ) ;
		return ( 0 ) ;
	}

 	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_copy_box_v2 <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top, tbl_size, edge_x, edge_y, obxyz_size, nbxyz_size,
		d_cp, blk_in_x, blk_in_y ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 

	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		temp_cube[i].x -= edge_x * 2 ;
		temp_cube[i].y -= edge_y * 2 ;
	}

	dbg_p_d_data_i_mn_v2("copy_box_v2 done", top, ( tbl_size / obxyz_size ) *
		nbxyz_size, 100, temp_cube, blk_in_x, blk_in_y ) ;
#endif 

	return ( 1 ) ;
}
