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

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_decode_misc.h"
#include "cs_sparser.h"
#include "cs_dct.h"
#include "cs_matrix.h"

#define CUDA_DBG


#define SIZE_V		5
#define SIZE_H		3
#define SIZE_T		2
#define SIZE_C		3

#define BUF_SIZE ( SIZE_V * SIZE_H * SIZE_T * SIZE_C )

struct vhtc myvhtc = {
	SIZE_V,
	SIZE_H,
	SIZE_T,
	SIZE_C,

	BUF_SIZE,
	0,
	BUF_SIZE * 2,

	0,
	0
} ;

float dbuf[ BUF_SIZE * 10 ] ;
float *dp1, *dp2, *dp3 ;
float *dp4, *dp5, *dp6, *dp7, *dp8 ;

int
main( int ac, char *av[] )
{
	int k ;
	float *dp ;
	float i ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	if (!h_do_dct_init())
	{
		printf("dct init failed \n") ;
		exit( 1 ) ;
	}

	myvhtc.size_mod2 = max_log2 ( BUF_SIZE ) ;
	myvhtc.A_size = BUF_SIZE ;
	myvhtc.D_size = BUF_SIZE * 2 ;

#ifdef CUDA_DBG 
	printf("myvhtc: v %d h %d t %d c %d - size %d %d %d -- A/D %d %d \n",
		myvhtc.v,
		myvhtc.h,
		myvhtc.t,
		myvhtc.c,
		myvhtc.size,
		myvhtc.size_mod2,
		myvhtc.size_x2,
		myvhtc.A_size,
		myvhtc.D_size ) ;
#endif 

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &dp1, sizeof ( float ) * BUF_SIZE * 18 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	dp2 = dp1 + ( BUF_SIZE ) ;
	dp3 = dp2 + ( BUF_SIZE ) ;
	dp4 = dp3 + ( BUF_SIZE ) * 2 ;
	dp5 = dp4 + ( BUF_SIZE ) * 2 ;
	dp6 = dp5 + ( BUF_SIZE ) * 2 ;
	dp7 = dp6 + ( BUF_SIZE ) * 2 ;
	dp8 = dp7 + ( BUF_SIZE ) * 2 ;

	printf("dp ::: \n1 %p \n2 %p \n3 %p \n4 %p \n5 %p \n6 %p \n7 %p \n8 %p\n",
		dp1, dp2, dp3, dp4, dp5, dp6, dp7, dp8 ) ;

	dp = dbuf ;

	// testing of h_compSprsVec

	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;

	if (!h_compSprsVec( dp1, dp2, dp3, &myvhtc, BUF_SIZE ))
	{
		printf("%s: h_compSprsVec failed\n", __func__ ) ;
		exit ( 0 ) ;
	}

	dbg_p_d_data_f_mn ( "h_compSprsVec dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec v", dp3, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec h", dp3 + BUF_SIZE, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;

	// testing of h_compSprsVecTrnsp
	// need to turn on the debug in cs_sparser.cu:h_compSprsVecTrnsp

	if (!h_compSprsVecTrnsp( dp3, dp1, dp2, BUF_SIZE * 2, &myvhtc ))
	{
		printf("%s: h_compSprsVecTrnsp failed\n", __func__ ) ;
		exit ( 0 ) ;
	}

	// testing _v and _h logic

	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;

	h_compDiffTrnspPx1_v( dp1, dp2, BUF_SIZE, &myvhtc ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	h_compDiffTrnspPx1_h( dp1, dp2, BUF_SIZE, &myvhtc ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	h_compDiffInside_v( dp1, dp2, BUF_SIZE, &myvhtc ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "inside v dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "inside v dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	h_do_cleanup_total_value_vh_elements( dp2, &myvhtc, 1 ) ;
#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "after total value v", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	h_compDiffInside_h( dp1, dp2, BUF_SIZE, &myvhtc ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "inside h dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "inside h dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	h_do_cleanup_total_value_vh_elements( dp2, &myvhtc, 0 ) ;
#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "after total value h", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	// if ( h_do_dct( dp1, dp2, SIZE_H, SIZE_V, SIZE_T ))
	if ( h_do_dct( dp1, dp2, 3, 5, 3, 0 )) 	// not inverse
	{
#ifdef CUDA_DBG 
		dbg_p_d_data_f_mn ( "before dct dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
		dbg_p_d_data_f_mn ( "after dct dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 
	} else
	{
		printf("%s: h_do_dct failed \n", __func__ ) ;
		exit( 2 ) ;
	}

	// testing of optimize()

	dp = dbuf ;

	i = -10.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i + 2 * k ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;

	dp = dbuf ;

	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i + 2 * k ;

	put_d_data_f( dp2, dbuf, sizeof( float ) * BUF_SIZE ) ;

	dbg_p_d_data_f("before optimize dp1", dp1, BUF_SIZE ) ;
	dbg_p_d_data_f("before optimize dp2", dp2, BUF_SIZE ) ;
	dbg_p_d_data_f("before optimize dp3", dp3, BUF_SIZE ) ;

	h_optimize( dp1, dp2, dp3, 0.2, BUF_SIZE ) ;

	dbg_p_d_data_f("after optimize", dp2, BUF_SIZE ) ;
	dbg_p_d_data_f("after optimize dp3", dp3, BUF_SIZE ) ;

	// testing of h_do_Jopt

	dp = dbuf ;

	i = -10.1 ;
	for ( k = 0 ; k < ( BUF_SIZE * 2 ) ; k++ )
		*dp++ = i + 2 * k ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * ( BUF_SIZE * 2 )) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("before Jopt", dp1, BUF_SIZE * 2 ) ;
#endif 

	i = h_do_Jopt( dp1, dp2, &myvhtc ) ; 

#ifdef CUDA_DBG 
	dbg_p_d_data_f("after Jopt", dp2, BUF_SIZE * 2 ) ;
	printf("Jopt %f .. \n", i ) ;
#endif 

	// testing of h_optimize_solver_w

	dp = dbuf ;

	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;

	dp = dbuf ;

	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE * 2 ; k++ )
		*dp++ = i + 2 * k ;

	put_d_data_f( dp3, dbuf, sizeof( float ) * BUF_SIZE * 2 ) ;

	h_optimize_solver_w( dp1, 0.2, dp3, NULL, &i, dp5, dp6, dp7, dp8, &myvhtc ) ;

#ifdef CUDA_DBG 
	printf("after h_optimize_solver_w Jopt %f .. \n", i ) ;
	dbg_p_d_data_f_mn ( "h_optimize_solver_w dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_optimize_solver_w dp3", dp3, BUF_SIZE * 2, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_optimize_solver_w dp5", dp5, BUF_SIZE * 2, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_optimize_solver_w dp6", dp6, BUF_SIZE * 2, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_optimize_solver_w dp8", dp8, BUF_SIZE * 2, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

}
