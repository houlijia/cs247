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
#include "cs_random.h"
#include "cs_buffer.h"
#include "cs_perm_generic.h"
#include "cs_vector.h"
#include "cs_decode_misc.h"
#include "cs_sparser.h"
#include "cs_dct.h"
#include "cs_matrix.h"
#include "cs_compGrad_x.h"

#define CUDA_DBG


#define SIZE_V		5
#define SIZE_H		3
#define SIZE_T		2
#define SIZE_C		3

#define NUM_MEASUREMENTS	128 
#define NUM_MEASUREMENTS_20	20 

struct vhtc myvhtc = {
	SIZE_V,
	SIZE_H,
	SIZE_T,
	SIZE_C,
	( SIZE_V * SIZE_H * SIZE_T * SIZE_C )
} ;

#define O_SIZE	10

#define BUF_SIZE ( SIZE_V * SIZE_H * SIZE_T * SIZE_C )

int dbuf_i[ BUF_SIZE * 10 ] ;
float dbuf[ BUF_SIZE * 10 ] ;
float *dp1, *dp2, *dp3, *dp5, *dp6 ;

// buf_index

enum buffer_index {
	BUFFER_VHTC,
	BUFFER_VHTC_MOD2,
	BUFFER_VHTC_x2,
	BUFFER_MAX_IDX
} ;

#define BUF_LOG_2		128 // 5x3x2*3 == 90 --> 128 * 4 == 512

struct cs_buf_desc my_buf[] = {
	{ BUF_SIZE * sizeof( float ), 7 },
	{ BUF_LOG_2 * sizeof( float ), 4 },
	{ BUF_SIZE * sizeof( float ) * 2, 4 }
} ;

RndC_uint32 *d_pL, *d_pR ;

struct grad_x_const grad_x_const ;
struct Pgrad_x Pgrad_x ;
struct beta beta ;
struct cjerr cjerr ;
struct LRperms LRperms ;

float *d_grad_xp ;
int *d_zerop ;
int d_zero_cnt = 2 ;

int
main( int ac, char *av[] )
{
	int k ;
	float gLg, *dp ;
	float i ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	if (!h_do_dct_init())
	{
		printf("dct init failed \n") ;
		exit( 1 ) ;
	}

	beta.A = 0.2 ;
	beta.D = 0.3 ;
	// beta.final = 0.5 ;
	beta.scldA = 0.4 ;

	dbg_init ( 1024 * 1024 ) ;

	if ( !cs_buffer_init ( my_buf, BUFFER_MAX_IDX ))
	{
		printf("ERR: cs_buffer_init failed \n") ;
		exit( 20 ) ;
	}

	// set up random numbers buffer 

	if (( i = cudaMalloc ( &d_pL, BUF_LOG_2 * sizeof( int ) * 2  )) != cudaSuccess )
	{
#ifdef CUDA_DBG 
		printf("%s: malloc failed %d \n", __func__, i ) ;
#endif 
		exit ( 3 ) ;
	}

	d_pR = d_pL + BUF_LOG_2 ;

	h_set_random_table( 0, NULL, d_pL, BUF_LOG_2, 1, 1 ) ;
	h_set_random_table( 1, NULL, d_pR, BUF_LOG_2, 0, 1 ) ;

	dbg_p_d_data_i ( "d_pL", ( int *)d_pL, BUF_LOG_2 ) ;
	dbg_p_d_data_i ( "d_pR", ( int *)d_pR, BUF_LOG_2 ) ;

	LRperms.d_Lperm = d_pL ;
	LRperms.d_Rperm = d_pR ;

	printf("testing of h_do_multVec/h_do_multTrnspVec -- with no zeroed rows ----------------------\n") ;

	// set up the zeroed rows

	d_zerop = ( int *)cs_get_free_list( BUFFER_VHTC ) ;
	dbuf_i[0] = 14 ;
	dbuf_i[1] = 16 ;

	d_zero_cnt = 2 ;

	put_d_data_i( d_zerop, dbuf_i, sizeof( int ) * 2 ) ;

	// set up the measurement

	dp = dbuf ;
	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	dp1 = ( float *)cs_get_free_list( BUFFER_VHTC_MOD2 ) ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "p1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	dp2 = ( float *)cs_get_free_list( BUFFER_VHTC_MOD2 ) ;
	dp3 = ( float *)cs_get_free_list( BUFFER_VHTC_MOD2 ) ;

	// no zeroed rows

	h_do_multVec( dp1, BUF_SIZE, dp2, NUM_MEASUREMENTS, dp3, BUF_LOG_2,
		( RndC_uint32 *)d_pL, ( RndC_uint32 *)d_pR, 
		NULL, 0 ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("after h_do_multVec no zero", dp2, BUF_LOG_2 ) ;
#endif 

	// testing of h_do_multTrnspVec

	h_do_multTrnspVec( dp2, dp1,
		( RndC_uint32 * )d_pL, ( RndC_uint32 * ) d_pR, 
		NUM_MEASUREMENTS, BUF_LOG_2, BUF_SIZE ) ;

	// NOTE ::: should see the orig data here ...

	// with zeroed rows 

	printf("testing of h_do_multVec/h_do_multTrnspVec -- with zeroed rows ----------------------\n") ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("after h_do_multVec no zero", dp2, BUF_LOG_2 ) ;
#endif 

	h_do_multVec( dp1, BUF_SIZE, dp2, NUM_MEASUREMENTS, dp3, BUF_LOG_2,
		( RndC_uint32 *)d_pL, ( RndC_uint32 *)d_pR, 
		d_zerop, d_zero_cnt ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("after h_do_multVec with 14, 16 zeroed rows", dp2, BUF_LOG_2 ) ;
#endif 

	h_do_multTrnspVec( dp2, dp1,
		( RndC_uint32 * )d_pL, ( RndC_uint32 * ) d_pR, 
		NUM_MEASUREMENTS, BUF_LOG_2, BUF_SIZE ) ;
	
#ifdef CUDA_OBS 
	// orig data
	cs_put_free_list (( char *)dp1, BUFFER_VHTC_MOD2 ) ;
#endif 
	cs_put_free_list (( char *)dp2, BUFFER_VHTC_MOD2 ) ;
	cs_put_free_list (( char *)dp3, BUFFER_VHTC_MOD2 ) ;

	printf("testing of h_do_Grad_x ---------------------------------------------------------\n") ;
 
	dp1 = ( float *)cs_get_free_list( BUFFER_VHTC ) ;
	dp2 = ( float *)cs_get_free_list( BUFFER_VHTC ) ;
	dp3 = ( float *)cs_get_free_list( BUFFER_VHTC ) ;
	d_grad_xp = ( float *)cs_get_free_list( BUFFER_VHTC_MOD2 ) ;
	dp5 = ( float *)cs_get_free_list( BUFFER_VHTC ) ;
	dp6 = ( float *)cs_get_free_list( BUFFER_VHTC ) ;

	dp = dbuf ;
	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;

	dp = dbuf ;
	i = 0.2 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp2, dbuf, sizeof( float ) * BUF_SIZE ) ;

	cjerr.d_A = dp1 ;
	cjerr.d_D = dp2 ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("cjerr.d_A", ( float *)dp1, BUF_SIZE ) ;
	dbg_p_d_data_f ("cjerr.d_D", ( float *)dp2, BUF_SIZE ) ;
#endif


	dp = dbuf ;
	i = 0.3 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp3, dbuf, sizeof( float ) * BUF_SIZE ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("grad_x_const", ( float *)dp3, BUF_SIZE ) ;
#endif

	grad_x_const.d_sum = dp3 ;

	h_do_Grad_x( &beta, &cjerr, &grad_x_const,
		( float * )d_grad_xp, dp5, dp6, BUF_SIZE ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("grad_x", ( float *)d_grad_xp, BUF_SIZE ) ;
#endif

#ifdef CUDA_OBS 

	// use it for h_do_updateGrad_x
	// cs_put_free_list (( char * )dp1, BUFFER_VHTC ) ;
	// cs_put_free_list (( char * )dp2, BUFFER_VHTC ) ;
	// cs_put_free_list (( char * )dp3, BUFFER_VHTC ) ;
#endif 
	cs_put_free_list (( char * )dp5, BUFFER_VHTC ) ;
	cs_put_free_list (( char * )dp6, BUFFER_VHTC ) ;

	printf("testing of h_do_Pgrad_x no zero -----------------------------------------------------\n") ;
 
	dp2 = ( float *)cs_get_free_list( BUFFER_VHTC_MOD2 ) ;
	dp3 = ( float *)cs_get_free_list( BUFFER_VHTC_MOD2 ) ; // for tmp
	dp6 = ( float *)cs_get_free_list( BUFFER_VHTC_x2 ) ;

#ifdef CUDA_OBS 
	dp = dbuf ;
	i = 0.1 ;
	for ( k = 0 ; k < BUF_SIZE ; k++ )
		*dp++ = i++ ;

	put_d_data_f( dp1, dbuf, sizeof( float ) * BUF_SIZE ) ;
#endif 

	Pgrad_x.d_A = dp2 ;
	Pgrad_x.A_size = BUF_SIZE ;
	Pgrad_x.d_D = dp6 ;
	Pgrad_x.D_size = BUF_SIZE * 2 ;

	// no zeroed rows ...

	h_do_Pgrad_x( d_grad_xp, BUF_SIZE, &Pgrad_x,
		&LRperms,
		NULL, 0,
		dp3, BUF_LOG_2, 
		&myvhtc ) ;

	dbg_p_d_data_f_mn ( "h_compSprsVec dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;

	dbg_p_d_data_f_mn ( "h_compSprsVec dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec dp3", dp3, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec v", dp6, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec h", dp6 + BUF_SIZE, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
							
	// zeroed rows ...

	printf("testing of h_do_Pgrad_x two zeros -----------------------------------------------------\n") ;

	h_do_Pgrad_x( d_grad_xp, BUF_SIZE, &Pgrad_x,
		&LRperms,
		d_zerop, d_zero_cnt,
		dp3, BUF_LOG_2, 
		&myvhtc ) ;

	dbg_p_d_data_f_mn ( "h_compSprsVec dp2", dp2, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;

	dbg_p_d_data_f_mn ( "h_compSprsVec dp1", dp1, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec dp3", dp3, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec v", dp6, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec h", dp6 + BUF_SIZE, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
							
	cs_put_free_list(( char *)dp1, BUFFER_VHTC_MOD2 ) ;
	cs_put_free_list(( char *)dp3, BUFFER_VHTC_MOD2 ) ;

	printf("testing of h_do_gLg -----------------------------------------------------\n") ;

	dp5 = ( float *)cs_get_free_list( BUFFER_VHTC_x2 ) ;

#ifdef CUDA_DBG 
	printf("beta D %f beta.scldA %f \n", beta.D, beta.scldA ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec Pgrad_x.D", Pgrad_x.d_D, BUF_SIZE * 2, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec Pgrad_x.A", Pgrad_x.d_A, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	gLg = h_do_gLg ( &beta, &Pgrad_x, dp5, &myvhtc ) ;

	printf("h_do_gLg ::: gLg %f \n", gLg ) ;

	cs_put_free_list(( char *)dp5, BUFFER_VHTC_x2 ) ;

	p_buffer_dbg("before h_updateGrad_x") ;

	printf("testing of h_updateGrad_x -----------------------------------------------------\n") ;

	dp1 = ( float *)cs_get_free_list( BUFFER_VHTC_x2 ) ;
	dp3 = ( float *)cs_get_free_list( BUFFER_VHTC_x2 ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ("cjerr.d_A", ( float *)cjerr.d_A, BUF_SIZE ) ;
	dbg_p_d_data_f ("cjerr.d_D", ( float *)cjerr.d_D, BUF_SIZE ) ;
	dbg_p_d_data_f ("grad_x_const", ( float *)grad_x_const.d_sum, BUF_SIZE ) ;
#endif

	gLg = h_updateGrad_x ( &beta, &cjerr, &grad_x_const, 
		d_grad_xp, dp1, dp3, BUF_LOG_2,
		&Pgrad_x,
		&LRperms, 
		d_zerop, d_zero_cnt,
		&myvhtc ) ;

#ifdef CUDA_DBG 
	printf("beta D %f beta.scldA %f \n", beta.D, beta.scldA ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec Pgrad_x.D", Pgrad_x.d_D, BUF_SIZE * 2, SIZE_H, SIZE_V, SIZE_H ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec Pgrad_x.A", Pgrad_x.d_A, BUF_SIZE, SIZE_H, SIZE_V, SIZE_H ) ;
#endif 

	printf("h_updateGrad_x ::: gLg %f \n", gLg ) ;

	cs_put_free_list(( char *)dp1, BUFFER_VHTC ) ;
	cs_put_free_list(( char *)dp3, BUFFER_VHTC ) ;

}
