#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_header.h"
#include "cs_copy_box.h"
#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_motion_detect_v4.h"
#include "cs_analysis.h"
#include "cs_mean_sd.h"

#define CUDA_DBG 

// total is actually blk_in_y * blk_in_x
__global__ void
d_mean_sd_divide ( float *dp, struct cube *dcubep, int blk_in_x, int blk_in_y, int total )
{
	int i, ot_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ( ot_idx < total )
	{
		i = get_blk_type_idx ( ot_idx, blk_in_x, blk_in_y ) ;
		i = dcubep[i].md_v4_record_length ;

		dp[ ot_idx ] /= i ;
	}
}	

// total is actually blk_in_y * blk_in_x
__global__ void
d_mean_sd_sqrt ( float *dp, int total )
{
	int ot_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ( ot_idx < total )
		dp[ ot_idx ] = sqrt( dp[ ot_idx ] ) ;
}	

// mean and sd points to blk_in_x * blk_in_y * sizeof ( float ) each 
// total is actually blk_in_y * blk_in_x
__global__ void
d_mean_sd_sd1 ( float *odp, struct cube *dcubep, float *d_meanp, int blk_in_x,
	int blk_in_y, int max_in_blk, int blk_size, int total )
{
	int i, blk_idx, t_idx, ot_idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *dp ;

	while ( ot_idx < total )
	{
		t_idx = ot_idx ;

		blk_idx = t_idx / max_in_blk ;	 // blk idx
		i = get_blk_type_idx ( blk_idx, blk_in_x, blk_in_y ) ;
		i = dcubep[i].md_v4_record_length ;

		t_idx %= max_in_blk ;
		if ( t_idx < i )
		{
			dp = odp + blk_idx * blk_size ;	// dp points to beginning of block 

			dp[ t_idx ] = (( dp[ t_idx ] - d_meanp[ blk_idx ]) * 
				( dp[ t_idx ] - d_meanp[ blk_idx ])) / ( i - 1 ) ;
		}

		ot_idx += CUDA_MAX_THREADS ;
	}
}	

// mean and sd points to blk_in_x * blk_in_y * sizeof ( float ) each 
int
h_mean_sd ( float *dp, float *tp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size, float *d_meanp, float *d_sdp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks, i ;
	int max_cnt ;
	struct cube cxyz[ CUBE_INFO_CNT ] ;

	memcpy ( cxyz, hcubep, sizeof ( *hcubep ) * CUBE_INFO_CNT ) ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: dp %p tp %p blksize %d blk x/y %d %d\n", __func__, dp, tp,
		blk_size, blk_in_x, blk_in_y ) ;
#endif 

	max_cnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
#ifdef CUDA_OBS 
		printf("%s :: cube before loop %d leng %d size %d \n", __func__, 
			cxyz[i].md_v4_loopcnt, cxyz[i].md_v4_record_length, cxyz[i].size ) ;
#endif 

		cxyz[i].md_v4_loopcnt = 1 ;
		cxyz[i].md_v4_record_length = cxyz[i].size ;

		if ( cxyz[i].size > max_cnt )
			max_cnt = cxyz[i].size ;

#ifdef CUDA_OBS 
		printf("%s :: cube after loop %d leng %d size %d \n", __func__, 
			cxyz[i].md_v4_loopcnt, cxyz[i].md_v4_record_length, cxyz[i].size ) ;
#endif 
	}

	// save a copy from dp to tp
	i = blk_size * blk_in_x * blk_in_y ;

	h_do_copy_vec( dp, tp, i, blk_size, blk_size ) ;

	// get the sum
	h_do_l1_norm_step2_v4( tp, cxyz, d_cubep, blk_in_x, blk_in_y, blk_size, 0 ) ;

	// copy the sum to d_meanp
	h_do_copy_vec<float> ( tp, d_meanp, blk_in_x * blk_in_y, blk_size, 1 ) ;

	// divede the sum with size to get mean

	i = blk_in_y * blk_in_x ;
	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_mean_sd_divide <<< nBlocks, nThreadsPerBlock >>> ( d_meanp, d_cubep, blk_in_x, blk_in_y, i ) ;

	cudaThreadSynchronize() ;

	// ok get the sd ...

	// save a copy from dp to tp
	i = blk_size * blk_in_x * blk_in_y ;

	h_do_copy_vec<float> ( dp, tp, i, blk_size, blk_size ) ;

	// do the ( x - u )^2/(n-1)
	i = max_cnt * blk_in_y * blk_in_x ;
	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_mean_sd_sd1<<< nBlocks, nThreadsPerBlock >>> ( tp, d_cubep, d_meanp,
		blk_in_x, blk_in_y, max_cnt, blk_size, i ) ;

	cudaThreadSynchronize() ;

	// do the sum again ...
	h_do_l1_norm_step2_v4( tp, cxyz, d_cubep, blk_in_x, blk_in_y, blk_size, 0 ) ;

	// copy the sum to sd 
	h_do_copy_vec<float> ( tp, d_sdp, blk_in_x * blk_in_y, blk_size, 1 ) ;

	// do sqrt the to the sum's

	i = blk_in_y * blk_in_x ;
	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_mean_sd_sqrt <<< nBlocks, nThreadsPerBlock >>> ( d_sdp, i ) ;

	cudaThreadSynchronize() ;

	h_set_cube_config ( d_cubep, hcubep ) ; 

	return ( 1 ) ;
}


