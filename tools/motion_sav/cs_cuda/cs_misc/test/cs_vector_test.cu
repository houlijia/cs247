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
#include "cs_vector.h"

#define CUDA_DBG
#define BUF_SIZE_INT		10

int *dp1, *dp2, *dp3 ;
float *fp1 ;

int buf1[ BUF_SIZE_INT ] ; 
float dbuf[ BUF_SIZE_INT ] ;

int
main( int ac, char *av[] )
{
	int k ;
	float *dbp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &dp1, sizeof ( int ) * BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	if (( k = cudaMalloc( &fp1, sizeof ( float ) * BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	// test h_do_vector_zero_some : 

	dbp = dbuf ;

	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
	{
		*dbp++ = ( float )k * 1.111 ;
	}

	buf1[0] = 1 ;
	buf1[1] = 5 ;
	buf1[2] = 9 ;

	put_d_data_i( dp1, buf1, sizeof( int ) * 3 ) ;
	dbg_p_d_data_i("before idx", dp1, 3 ) ;

	cudaMemcpy( fp1, dbuf, sizeof( float ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	dbg_p_d_data_f("before h_do_vector_zero_some", fp1, BUF_SIZE_INT ) ;

	h_do_vector_zero_some ( fp1, dp1, 3 ) ;

	dbg_p_d_data_i("after idx", dp1, 3 ) ;
	dbg_p_d_data_f("after h_do_vector_zero_some", fp1, BUF_SIZE_INT ) ;

}
