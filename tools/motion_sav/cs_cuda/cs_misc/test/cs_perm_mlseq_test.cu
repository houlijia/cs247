#include <stdio.h>

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_perm_mlseq.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_random.h"

#define TEST_SIZE 16
#define BUF_LOG_2  10
#define PERM_SIZE 8

int orig[] = { 11, 22, 33, 44, 55, 66, 77, 88, 88, 77, 66, 55, 44, 33, 22, 11 } ;
float orig1[] = { 11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0,
	88.8, 77.7, 66.6, 55.5, 44.4, 33.3, 22.2, 11.1 } ;
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
	put_d_data_i ( pp, perm, sizeof(int) * PERM_SIZE ) ;

	h_do_permutation_R<int> ( op, tp, pp, TEST_SIZE, PERM_SIZE ) ;

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
	   8 -- 00000021 33
	   9 -- 00000058 88
	   10 -- 00000016 22
	   11 -- 00000042 66
	   12 -- 0000004d 77
	   13 -- 00000037 55
	   14 -- 0000000b 11
	   15 -- 0000002c 44
	*/

	put_d_data_f (( float *)op, orig1, sizeof(float) * TEST_SIZE ) ;

	h_do_permutation_R<float> ( (float*)op, (float*)tp, pp, TEST_SIZE, PERM_SIZE ) ;

	dbg_p_d_data_f ("f2", ( float *)tp, TEST_SIZE ) ; 

	/* should see
	   0 -- 66.000000
	   1 -- 11.000000
	   2 -- 77.000000
	   3 -- 33.000000
	   4 -- 22.000000
	   5 -- 44.000000
	   6 -- 88.000000
	   7 -- 55.000000
	   8 -- 33.299999
	   9 -- 88.800003
	   10 -- 22.200001
	   11 -- 66.599998
	   12 -- 77.699997
	   13 -- 55.500000
	   14 -- 11.100000
	   15 -- 44.400002

	 */
}
