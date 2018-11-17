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

#define CUDA_DBG
#define BUF_SIZE_INT		10

int *dp1, *dp2, *dp3 ;
float *fp1 ;

int buf1[ BUF_SIZE_INT ] ; 
float dbuf[ BUF_SIZE_INT ] ;

char cbuf1[ BUF_SIZE_INT ] ;

/*
   exptected output ...

	h_do_vector_add_destroy: 45 
	h_do_vector_add_destroy: 36 
	h_do_dot: 4785 
	h_do_dot: 3804 

*/

int
main( int ac, char *av[] )
{
	int i, k, *dp ;
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

	// h_expand_c_to_i testing 

	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		cbuf1[k] = 0x7b + k ;

	cudaMemcpy( dp1, cbuf1, BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	h_cast_T_to_T<char, int> (( char *)dp1, ( int * )fp1, BUF_SIZE_INT ) ;
	dbg_p_d_data_i ("h_cast_T_to_T : char ",( int * )fp1, BUF_SIZE_INT );

	h_cast_T_to_T<unsigned char, int>(( unsigned char *)dp1, ( int * )fp1, BUF_SIZE_INT ) ;
	dbg_p_d_data_i ("h_cast_T_to_T : unsigned char ",( int * )fp1, BUF_SIZE_INT );

	h_cast_T_to_T<unsigned char, float>(( unsigned char *)dp1, ( float * )fp1, BUF_SIZE_INT ) ;
	dbg_p_d_data_f ("h_cast_T_to_T : to float",( float * )fp1, BUF_SIZE_INT );

	// test h_do_int_to_float : to diff buf

	dp = buf1 ;

	i = 0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	h_do_int_to_float ( dp1, fp1, BUF_SIZE_INT ) ;

	cudaMemcpy( dbuf, fp1, sizeof( float ) * BUF_SIZE_INT, cudaMemcpyDeviceToHost ) ;

	dbp = dbuf ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		printf("diff buf -- %d : %f \n", k, *dbp++ ) ;

	// test h_do_int_to_float : to same buf

	memset( dbuf, 0, BUF_SIZE_INT * sizeof ( float )) ; // since int and float are of the same size

	dbp = dbuf ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		printf("same buf -- %d : %f \n", k, *dbp++ ) ;

	dp = buf1 ;

	i = 4 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	h_do_int_to_float ( dp1, (float *)dp1, BUF_SIZE_INT ) ;

	cudaMemcpy( dbuf, dp1, sizeof( float ) * BUF_SIZE_INT, cudaMemcpyDeviceToHost ) ;

	dbp = dbuf ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		printf("same buf -- %d : %f \n", k, *dbp++ ) ;
}
