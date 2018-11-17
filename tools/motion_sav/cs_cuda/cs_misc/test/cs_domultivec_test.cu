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
#include "cs_domultivec.h"

#define CUDA_DBG

#define TST_SIZE 8

float *dp3, *dp1, *dp2, hd[ TST_SIZE ] ;
RndC_uint32 *k1, *k2 ;

int
main( int ac, char *av[] )
{
	int k ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &dp1, sizeof ( float ) * TST_SIZE * 3 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 9 ) ;
	}

	if (( k = cudaMalloc( &k1, sizeof ( RndC_uint32 ) * TST_SIZE * 2 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 9 ) ;
	}

	k2 = k1 + TST_SIZE ;
	dp2 = dp1 + TST_SIZE ;
	dp3 = dp2 + TST_SIZE ;

	for ( k = 0 ; k < TST_SIZE ; k++ )
		hd[k] = 0.5 * k + 1.0 ;

	if (( k = cudaMemcpy( dp1, hd, sizeof ( float ) * TST_SIZE,
		cudaMemcpyHostToDevice)) != cudaSuccess )
	{
		printf("%s: download cp fail: %d\n", __func__, k ) ;
		exit ( 1 ) ;
	}

	dbg_p_d_data_f("before do", dp1, TST_SIZE ) ;  

	if (!h_set_random_table( 0, NULL, k1, TST_SIZE, 0, 1 ))
	{
		printf("%s: k1 fail: \n", __func__ ) ;
		exit ( 1 ) ;
	}

	dbg_p_d_data_i("k1", ( int * )k1, TST_SIZE ) ;  

	if (!h_set_random_table( 1, NULL, k2, TST_SIZE, 0, 1 ))
	{
		printf("%s: k2 fail: \n", __func__ ) ;
		exit ( 1 ) ;
	}

	dbg_p_d_data_i("k2", ( int * )k2, TST_SIZE ) ;  

	h_do_multi_vec ( dp1, dp2, dp3, (int *)k1, (int *)k2, TST_SIZE - 3, -1 ) ;

	dbg_p_d_data_f("after do", dp2, TST_SIZE ) ;	// data in output

	h_do_multi_vec ( dp1, dp2, dp3, (int *)k1, (int *)k2, TST_SIZE - 3, 3 ) ;

	dbg_p_d_data_f("after do", dp2, TST_SIZE ) ;	// data in output
	// do multi transpose vec

	printf("------------------------------------------------------------------------\n") ;

	for ( k = 0 ; k < TST_SIZE ; k++ )
		hd[k] = 0.5 * k + 1.0 ;

	if (( k = cudaMemcpy( dp1, hd, sizeof ( float ) * TST_SIZE,
		cudaMemcpyHostToDevice)) != cudaSuccess )
	{
		printf("%s: download cp fail: %d\n", __func__, k ) ;
		exit ( 1 ) ;
	}

	dbg_p_d_data_f("TRANS:before do", dp1, TST_SIZE ) ;  

	dbg_p_d_data_i("TRANS k1", ( int * )k1, TST_SIZE ) ;  

	dbg_p_d_data_i("TRANS k2", ( int * )k2, TST_SIZE ) ;  

	h_do_multi_trnsp_vec( dp1, dp2, (int *)k1, (int *)k2, TST_SIZE ) ;

	dbg_p_d_data_f("TRANS after do", dp1, TST_SIZE ) ;	// data in input  

}
