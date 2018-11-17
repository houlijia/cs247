#include <stdio.h>

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_perm_generic.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_random.h"

#define TEST_SIZE 8
#define BUF_LOG_2  10

int orig[] = { 11, 22, 33, 44, 55, 66, 77, 88 } ;
float orig1[] = { 11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0 } ;
int perm[] = { 1, 4, 3, 5, 7, 0, 2, 6 } ;

int tst_2() ;

main()
{
	int k ;
	int *tp, *pp, *op ;
	float *ftp, *fpp, *fop ;

	dbg_init( 102400 ) ;

	if (( k = cudaMalloc( &pp, sizeof ( int ) * TEST_SIZE * 3 )) != cudaSuccess )
	{
		printf("%s: alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	op = pp + TEST_SIZE ;
	tp = op + TEST_SIZE ;

	put_d_data_i ( op, orig, sizeof(int) * TEST_SIZE ) ;
	put_d_data_i ( pp, perm, sizeof(int) * TEST_SIZE ) ;

	h_do_permutation_generic_f1 ( op, tp, pp, TEST_SIZE ) ;

	dbg_p_d_data_i ("f1", tp, TEST_SIZE ) ; 

	/* should see
	   0 -- 00000042 66
	   1 -- 0000000b 11
	   2 -- 0000004d 77
	   3 -- 00000021 33
	   4 -- 00000016 22
	   5 -- 0000002c 44
	   6 -- 00000058 88
	   7 -- 00000037 55
	*/

	h_do_permutation_generic_f2 ( op, tp, pp, TEST_SIZE ) ;

	dbg_p_d_data_i ("f2", tp, TEST_SIZE ) ; 

	/* should see
	   0 -- 00000016 22
	   1 -- 00000037 55
	   2 -- 0000002c 44
	   3 -- 00000042 66
	   4 -- 00000058 88
	   5 -- 0000000b 11
	   6 -- 00000021 33
	   7 -- 0000004d 77
	 */

	// float
	if (( k = cudaMalloc( &fpp, sizeof ( float ) * TEST_SIZE * 3 )) != cudaSuccess )
	{
		printf("%s: alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	fop = fpp + TEST_SIZE ;
	ftp = fop + TEST_SIZE ;

	put_d_data_f ( fop, orig1, sizeof(float) * TEST_SIZE ) ;
	put_d_data_i ( pp, perm, sizeof(int) * TEST_SIZE ) ;

	h_do_permutation_generic_f1 ( fop, ftp, pp, TEST_SIZE ) ;

	dbg_p_d_data_f ("float f1", ftp, TEST_SIZE ) ; 

	/* should see
	   0 -- 00000042 66
	   1 -- 0000000b 11
	   2 -- 0000004d 77
	   3 -- 00000021 33
	   4 -- 00000016 22
	   5 -- 0000002c 44
	   6 -- 00000058 88
	   7 -- 00000037 55
	*/

	h_do_permutation_generic_f2 ( fop, ftp, pp, TEST_SIZE ) ;

	dbg_p_d_data_f ("float f2", ftp, TEST_SIZE ) ; 

	/* should see
	   0 -- 00000016 22
	   1 -- 00000037 55
	   2 -- 0000002c 44
	   3 -- 00000042 66
	   4 -- 00000058 88
	   5 -- 0000000b 11
	   6 -- 00000021 33
	   7 -- 0000004d 77
	 */

	tst_2() ;

}

int
tst_2()
{
	// set up random numbers buffer

	int i, *t, *ot, *d_pL, *d_pR, *d_t1, *d_t2, *d_t3, *d_t4, *d_t5 ;

	printf("starting tst_2 ... \n") ;

	if (( i = cudaMalloc ( &d_pL, BUF_LOG_2 * sizeof( int ) * 7  )) != cudaSuccess )
	{
#ifdef CUDA_DBG 
		printf("%s: malloc failed %d \n", __func__, i ) ;
#endif 
		return ( 0 ) ;
	}

	d_pR = d_pL + BUF_LOG_2 ;
	d_t1 = d_pR + BUF_LOG_2 ;
	d_t2 = d_t1 + BUF_LOG_2 ;
	d_t3 = d_t2 + BUF_LOG_2 ;
	d_t4 = d_t3 + BUF_LOG_2 ;
	d_t5 = d_t4 + BUF_LOG_2 ;

	// set up orig data to be 0..9

	t = (int *)malloc ( BUF_LOG_2 * sizeof ( int )) ;

	ot = t ;

	for ( i = 0 ; i < BUF_LOG_2 ; i++ )
		*t++ = i ;

	put_d_data_i ( d_t1, ot, sizeof(int) * BUF_LOG_2 ) ;

	// set up random table

	h_set_random_table( 0, NULL, ( RndC_uint32 * )d_pL, BUF_LOG_2, 1, 1 ) ;
	h_set_random_table( 1, NULL, ( RndC_uint32 * )d_pR, BUF_LOG_2, 0, 1 ) ;

	dbg_p_d_data_i ( "d_pL", ( int *)d_pL, BUF_LOG_2 ) ;
	dbg_p_d_data_i ( "d_pR", ( int *)d_pR, BUF_LOG_2 ) ;

	// ok 

	dbg_p_d_data_i ( "d_t1", ( int *)d_t1, BUF_LOG_2 ) ;
	h_do_permutation_generic_f1( d_t1, d_t2, d_pR, BUF_LOG_2 ) ; 
	dbg_p_d_data_i ( "d_t2", ( int *)d_t2, BUF_LOG_2 ) ;

	h_do_permutation_generic_f2( d_t2, d_t3, d_pL, BUF_LOG_2 ) ; 
	dbg_p_d_data_i ( "d_t3", ( int *)d_t3, BUF_LOG_2 ) ;

	h_do_permutation_generic_f1( d_t3, d_t4, d_pL, BUF_LOG_2 ) ; 
	dbg_p_d_data_i ( "d_t4", ( int *)d_t4, BUF_LOG_2 ) ;

	h_do_permutation_generic_f2( d_t4, d_t2, d_pR, BUF_LOG_2 ) ; 
	dbg_p_d_data_i ( "d_t2", ( int *)d_t2, BUF_LOG_2 ) ;

	h_do_permutation_generic_inverse ( d_t5, d_pL, BUF_LOG_2 ) ;
	dbg_p_d_data_i ( "d_t5", ( int *)d_t5, BUF_LOG_2 ) ;

	return ( 1 ) ;
}
