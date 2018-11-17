#include <iostream>
using namespace std;

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_analysis.h"
#include "cs_header.h"
#include "cs_block.h"
#include "cs_perm_mlseq.h"
#include "cs_expand.h"
#include "cs_interpolate.h"
#include "cs_perm_selection.h"
#include "cs_copy_box.h"
#include "cs_motion_detect.h"
#include "cs_motion_detect_v2.h"
// #include "cs_edge_detect.h"
#include "cs_edge_detect_v2.h"
#include "cs_ipcam.h"

#define CUDA_DBG

int *dp1 = NULL, *dp2 = NULL ;
int *hp1 = NULL, *hp2 = NULL ;

#define NUM_OF_HVT_INDEX 3

#define BUF_SIZE	( 1024 * 1024 )
#define BUF_SIZE_INT	( BUF_SIZE * sizeof (int))

struct cs_xyz	hcube[ CUBE_INFO_CNT ], *dcubep ;
struct cube cubecube[ CUBE_INFO_CNT ] ;

int
main( int ac, char *av[] )
{
#ifdef CUDA_OBS
  int orig, rec_size, hvt_size;
#endif
	int k, i, *dp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	if (( k = cudaMalloc( &dcubep, sizeof ( cube ))) != cudaSuccess )
	{
		printf("%s: d_cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	if (( k = cudaMalloc( &dp1, BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	if (( k = cudaMalloc( &dp2, BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

#ifdef CUDA_OBS 
	// testing of the ipcam stuff
	ipcam_init ( 5, 640, 480 ) ;

	exit(2) ;
#endif 

	dbg_init( 1024 * 1024 ) ;
	
	cubecube[0].x = hcube[0].x = 12 ;
	cubecube[0].y = hcube[0].y = 14 ;
	cubecube[0].z = hcube[0].z = 3 ;
	cubecube[1].x = hcube[1].x = 10 ;
	cubecube[1].y = hcube[1].y = 8 ;
	cubecube[1].z = hcube[1].z = 3 ;
	cubecube[2].x = hcube[2].x = 8 ;
	cubecube[2].y = hcube[2].y = 6 ;
	cubecube[2].z = hcube[2].z = 4 ;

	h_set_config( dcubep, cubecube ) ;	

#ifdef CUDA_OBS 
	dbg_p_d_data_i ( "cube 1", ( int *)dcubep, sizeof( hcube ) / sizeof ( int)) ;
#endif 

	hp1 = ( int * )malloc ( BUF_SIZE_INT ) ;
	hp2 = ( int * )malloc ( BUF_SIZE_INT ) ;

	dp = hp1 ;
	for ( i = 0 ; i < BUF_SIZE ; i++ )
	{
		*dp++ = rand() & 0xff ;
#ifdef CUDA_OBS 
		*dp++ = i ;
		*dp++ = rand() & 0xff ;

		if ( i & 1 )
			*dp++ = i+100 ;
		else
			*dp++ = i-100 ;
#endif 
	}
	
	set_device_mem_i( dp2, BUF_SIZE, 111 ) ;

	if (( i = cudaMemcpy( dp1, hp1, BUF_SIZE_INT,
		cudaMemcpyHostToDevice)) != cudaSuccess )
	{
		printf("%s: cp failed %d\n", __func__, i ) ;
		exit( 0 ) ;
	}

#ifdef CUDA_DBG 
	dbg_p_d_data_i ( "dp1 original", dp1, 25 ) ;
#endif 

#ifdef CUDA_OBS 
	
	// test of dbg_p_d_data_i_mn_v2
	
	dbg_p_d_data_i_mn_v2( "dp1 init", dp1, 12 * 14 * 3 * 3 * 3, 12,
		hcube, 3 ,3 ) ;

#endif 


	// do the test here ..

#ifdef CUDA_OBS 
	// blocking 

	// not tested ... not tested ... not tested ...

	h_make_block( dp1, dp2,
		30, 16,	// x/y
		480,	// frame size ... x * y
		10, 8, 2	// block ... x/y/z
		0,	// no perm
		5, 4, // overlap ... x/y
		5, 3, // num of blocks in x/y
		2, 3, // append
		0, 0 ) ;	// no weight, no shift 

	exit( 23 ) ;
#endif 

#ifdef CUDA_OBS 
	// motion detection 
	
	k = h_do_motion_idx_v2 ( dp2, 1000 * 1000, &orig, 3, 3, cubecube,
		2, 1, 2, &rec_size ) ;

	if ( !k )
	{
		printf("motion failed") ;
		exit( 1 ) ;
	}

	printf("orig idx is %d size %d\n", orig, rec_size ) ;

	hvt_size = 5 * 3 * 2 ;		// ( 2 * 2 + 1 ) * ( 1 * 2 + 1 ) * 2

	dbg_p_d_data_i_mn ( "dp2", dp2, ( rec_size + NUM_OF_HVT_INDEX ) * ( hvt_size * 3 * 3 ), 
		rec_size + NUM_OF_HVT_INDEX, hvt_size * 3 * 3, 6 ) ;

	dbg_p_d_data_i ( "dp1 original", dp1, 30 ) ;

	// step 0 : copy data ...
	k = h_do_motion_detection_step0_v2 ( dp1, dp2,
		rec_size * hvt_size * 3 * 3, 
		rec_size,
		4, 2, 2,
		dcubep,
		hvt_size, cubecube[0].x * cubecube[0].y * cubecube[0].z  ) ;

	dbg_p_d_data_i ( "dp1 after", dp1, 30 ) ;

	dbg_p_d_data_i_mn ( "motion 1", dp1, 12 * 14 * 3 * 9, 12, 14, 12 ) ;
	dbg_p_d_data_i_mn ( "motion 2", dp2, ( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * 3 * 3, 
		rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

	// do step 1

	for ( k = 0 ; k < CUBE_INFO_CNT ; k++ )
	{
		cubecube[k].x -= 4 ;
		cubecube[k].y -= 2 ;
		cubecube[k].z -= 1 ;
	}

	h_set_config ( dcubep, cubecube ) ;

	h_do_l1_norm_step1_v2( dp2, rec_size * hvt_size * 3 * 3, rec_size, orig, hvt_size ) ;
	dbg_p_d_data_i_mn ( "step 1", dp2, ( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * 3 * 3, 
		rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

	// step 2

	// step 2 l1_norm ...

	k = h_do_l1_norm_step2_v2( dp2, rec_size * hvt_size * 3 * 3, 
		rec_size, cubecube, dcubep) ;

	printf("k is %d\n", k ) ;

	dbg_p_d_data_i_mn ( "step 2", dp2, ( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * 3 * 3, 
		rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

	dbg_p_d_data_i ( "dcubep after step 2", ( int *)dcubep, 9 ) ;

	// step 3
	
	k = h_do_l1_norm_step3_v2( dp2, rec_size * hvt_size * 3 * 3, rec_size,
		orig, hvt_size ) ;

	printf("k is %d\n", k ) ;

	dbg_p_d_data_i_mn ( "step 3", dp2, ( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * 3 * 3, 
		rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

	// step 4

	k = h_do_l1_norm_step4_v2( dp2, rec_size * hvt_size * 3 * 3,  rec_size,
		orig, hvt_size, hp1 ) ;

	printf("k is %d\n", k ) ;

	dbg_p_d_data_i_mn ( "step 4", dp2, ( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * 3 * 3, 
		rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

	dbg_pdata_i( "return values", hp1, 3 * 3 * sizeof ( int )) ;

	exit( 23 ) ;
#endif 

#ifdef CUDA_OBS 
	// edge v2 test

	{
		int edge_x = 1, edge_y = 2, blk_in_x = 4, blk_in_y = 3 ;

		i = blk_in_x * blk_in_y * (hcube[0].x * hcube[0].y * hcube[0].z) ;
		h_do_edge_detection_v2 ( dp1, dp2, i,
			dcubep, edge_x, edge_y, blk_in_x, blk_in_y,
			( struct cube * )&cubecube[0] ) ;

		dbg_p_d_data_i_mn_v2 ( "edge_v2 orig", dp1, i, hcube[0].x, hcube,
			blk_in_x, blk_in_y ) ;
		dbg_p_d_data_i_mn_v2 ( "edge_v2 get", dp2, i, hcube[0].x, hcube,
			blk_in_x, blk_in_y ) ;

		i = h_do_copy_box_v2 ( dp2, dp1, i, 
			edge_x, edge_y, blk_in_x, blk_in_y, dcubep, cubecube  ) ;
		dbg_p_d_data_i_mn ( "after copy", dp1, ( hcube[0].x - edge_x * 2 ) *
			( hcube[0].y - edge_y * 2 ) * hcube[0].z * blk_in_x * blk_in_y, 
			hcube[0].x - edge_x * 2, hcube[0].y - edge_y * 2, 
			hcube[0].x - edge_x * 2) ;
	}
#endif 



#ifdef CUDA_OBS 
	// do step 1
	dbg_p_d_data_i_mn ( "l1", dp1, ( 10 * 8 ), 10, 8, 10 ) ;
	h_do_l1_norm_step1( dp1, 100, 7, 4 ) ;
	dbg_p_d_data_i_mn ( "l1 done", dp1, ( 10 * 8 ), 10, 8, 10 ) ;

#endif 

#ifdef CUDA_OBS 

	// step 3 l1-norm

	dbg_p_d_data_i_mn ( "l3 before", dp1, ( 11 * 7 ), 11, 7, 11 ) ;
	k = h_do_l1_norm_step3( dp1, 7 * 8, 8, 3) ;
	printf("k is %d\n", k ) ;
	dbg_p_d_data_i_mn ( "l3 done", dp1, ( 11 * 7 ), 11, 7, 11 ) ;

	// step 4 find min

	dbg_p_d_data_i_mn ( "l4 before", dp1, ( 11 * 7 ), 11, 7, 11 ) ;
	k = h_do_l1_norm_step4( dp1, 8 * 7, 8 , 3, a ) ;
	printf("k is %d -- %d %d %d %d \n", k, a[0], a[1], a[2], a[3] ) ;
	dbg_p_d_data_i_mn ( "l4 done", dp1, ( 11 * 7 ), 11, 7, 11 ) ;

#endif 

#ifdef CUDA_OBS 

	// step 2 l1_norm ...

	dbg_p_d_data_i_mn ( "l2 before", dp1, ( 11 * 7 ), 11, 7, 11 ) ;
	k = h_do_l1_norm_step2( dp1, 7 * 8, 8) ;
	printf("k is %d\n", k ) ;
	dbg_p_d_data_i_mn ( "l2 done", dp1, ( 11 * 7 ), 11, 7, 11 ) ;
#endif 

#ifdef CUDA_OBS 

	//abs() test

	dbg_p_d_data_i_mn ( "abs", dp1, ( 13 * 8 ), 13, 8, 13 ) ;

	k = h_set_abs ( dp1, 100, 10, 3 ) ;

	printf("k is %d\n", k ) ;

	dbg_p_d_data_i_mn ( "abs done", dp1, ( 13 * 8 ), 13, 8, 13 ) ;
#endif 

#ifdef CUDA_OBS 

	// motion detection 
	
	k = h_do_motion_idx ( dp2, 100 * 100, 72,
		3, 3, 3, &i ) ;

	if ( !k )
	{
		printf("motion failed") ;
		exit( 1 ) ;
	}

	printf("orig idx is %d\n", i ) ;

	dbg_p_d_data_i_mn ( "dp1", dp1, (6*8*5), 6, 8, 6 ) ;

	k = h_do_motion_detection ( dp1, dp2, 
		72 * 27,
		75,	// 4 * 6 * 3
		6, 48,
		4, 24 ) ;

	dbg_p_d_data_i_mn ( "motion", dp2, ( 75 * 28 ), 75, 28, 75 ) ;
#endif 

#ifdef CUDA_OBS 
	k = h_do_copy_box ( dp1, dp2, 96, 8,6,1,1 ) ;

	if ( !k )
	{
		printf("copy failed") ;
		exit( 1 ) ;
	}

	dbg_p_d_data_i_mn ( "copy before", dp1, 128, 8, 6, 8 ) ;
	dbg_p_d_data_i_mn ( "copy done", dp2, 100, 6, 4, 6 ) ;

#endif 

}
