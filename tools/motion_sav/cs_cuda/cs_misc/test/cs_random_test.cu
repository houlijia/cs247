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

#define CUDA_DBG

// #define K64	(1024 * 64)
#define K64	1024

#define TST_SIZE 12

RndC_uint32 *dp1, Krandom[ K64 ], Krandom1[ K64 ] ;
RndC_uint32 seed = 1000 ;

#ifdef CUDA_OBS

// from matlab

>> s = RandStream('mt19937ar','Seed',1000);
>> RandStream.setGlobalStream(s);
>> randperm(11)

	ans =

	     7     2    11     6     9     8     4     1    10     5     3

#endif 

int
main( int ac, char *av[] )
{
	int k ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &dp1, sizeof ( RndC_uint32 ) * K64 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 9 ) ;
	}

	// no host buf

	if (!( k = h_set_random_table( seed, ( RndC_uint32 * )NULL, ( RndC_uint32 * )dp1, TST_SIZE, 1, 1 )))
	{
		fprintf( stderr, "1 failed\n") ;
		exit( 1 ) ;
	}

	dbg_p_d_data_i("after one", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after one", ( int * )dp1, TST_SIZE ) ;

	if (!( k = h_set_random_table( seed, ( RndC_uint32 * )NULL, ( RndC_uint32 * )dp1, TST_SIZE, 0, 1 )))
	{
		fprintf( stderr, "2 failed\n") ;
		exit( 2 ) ;
	}

	dbg_p_d_data_i("after two", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after two", ( int * )dp1, TST_SIZE ) ;

	// with host buf

	if (!( k = h_set_random_table( seed, Krandom, ( RndC_uint32 * )dp1, TST_SIZE, 1, 1 )))
	{
		fprintf( stderr, "3 failed\n") ;
		exit( 1 ) ;
	}

	dbg_pdata_i("after 3 host", ( int *)Krandom, TST_SIZE ) ;
	dbg_p_d_data_i("after three", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after three", ( int * )dp1, TST_SIZE ) ;

	if (!( k = h_set_random_table( seed, Krandom, ( RndC_uint32 * )dp1, TST_SIZE, 0, 1 )))
	{
		fprintf( stderr, "4 failed\n") ;
		exit( 2 ) ;
	}

	dbg_pdata_i("after 4 host", ( int *)Krandom, TST_SIZE ) ;
	dbg_p_d_data_i("after four", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after four", ( int * )dp1, TST_SIZE ) ;

	printf("only init once ......................................................\n") ;

	// no host buf

	if (!( k = h_set_random_table( seed, ( RndC_uint32 * )NULL, ( RndC_uint32 * )dp1, TST_SIZE, 1, 1 )))
	{
		fprintf( stderr, "1 failed\n") ;
		exit( 1 ) ;
	}

	dbg_p_d_data_i("after one", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after one", ( int * )dp1, TST_SIZE ) ;

	if (!( k = h_set_random_table( seed, ( RndC_uint32 * )NULL, ( RndC_uint32 * )dp1, TST_SIZE, 0, 0 )))
	{
		fprintf( stderr, "2 failed\n") ;
		exit( 2 ) ;
	}

	dbg_p_d_data_i("after two", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after two", ( int * )dp1, TST_SIZE ) ;

	// with host buf

	if (!( k = h_set_random_table( seed, Krandom, ( RndC_uint32 * )dp1, TST_SIZE, 1, 1 )))
	{
		fprintf( stderr, "3 failed\n") ;
		exit( 1 ) ;
	}

	dbg_pdata_i("after 3 host", ( int *)Krandom, TST_SIZE ) ;
	dbg_p_d_data_i("after three", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after three", ( int * )dp1, TST_SIZE ) ;

	if (!( k = h_set_random_table( seed, Krandom, ( RndC_uint32 * )dp1, TST_SIZE, 0, 0 )))
	{
		fprintf( stderr, "4 failed\n") ;
		exit( 2 ) ;
	}

	dbg_pdata_i("after 4 host", ( int *)Krandom, TST_SIZE ) ;
	dbg_p_d_data_i("after four", ( int * )dp1, TST_SIZE ) ;  
	dbg_ck_unique ("after four", ( int * )dp1, TST_SIZE ) ;
}
