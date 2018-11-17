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
#include "cs_dct.h"

#include "cs_whm_encode_b.h"

#define CUDA_DBG

int *dp1 = NULL ;
int *hp1 = NULL, *hp2 = NULL ;

#define NUM_OF_HVT_INDEX 3

#define BUF_SIZE	( 1024 * 1024 )
#define BUF_SIZE_INT	( BUF_SIZE * sizeof (int))

#define LOG2_SIZE	8	// has to be power of 2 

int buf1[LOG2_SIZE] ; 
float fbuf1[LOG2_SIZE] ; 

float *fp1 ;

int
main( int ac, char *av[] )
{
	int k, *fp, *dp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &dp1, BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	// integer ...

	dp = buf1 ;

	for ( k = 0 ; k < LOG2_SIZE ; k++ )
		*dp++ = k + 10 ;

	// normal whm

	dbg_put_d_data (( char *)dp1, ( char *)buf1, sizeof( int ) * LOG2_SIZE ) ;

	dbg_p_d_data_i ( "before whm", dp1, LOG2_SIZE ) ;

	cs_whm_measurement_b( dp1, LOG2_SIZE, LOG2_SIZE ) ;

	dbg_p_d_data_i ( "after whm", dp1, LOG2_SIZE ) ;

	// iwhm

	dbg_put_d_data (( char *)dp1, ( char *)buf1, sizeof( int ) * LOG2_SIZE ) ;

	dbg_p_d_data_i ( "before iwhm", dp1, LOG2_SIZE ) ;

	cs_iwhm_measurement_b( dp1, LOG2_SIZE, LOG2_SIZE ) ;

	dbg_p_d_data_i ( "after iwhm", dp1, LOG2_SIZE ) ;

	// float ...

	fp1 = fbuf1 ;

	for ( k = 0 ; k < LOG2_SIZE ; k++ )
		*fp1++ = (float )k + 10.2 ;

	// normal whm

	dbg_put_d_data (( char *)dp1, ( char *)fbuf1, sizeof( float ) * LOG2_SIZE ) ;

	dbg_p_d_data_f ( "before whm", ( float * )dp1, LOG2_SIZE ) ;

	cs_whm_measurement_b( ( float * )dp1, LOG2_SIZE, LOG2_SIZE ) ;

	dbg_p_d_data_f ( "after whm", ( float * )dp1, LOG2_SIZE ) ;

	// iwhm

	dbg_put_d_data (( char *)dp1, ( char *)fbuf1, sizeof( float ) * LOG2_SIZE ) ;

	dbg_p_d_data_f ( "before iwhm", ( float *)dp1, LOG2_SIZE ) ;

	cs_iwhm_measurement_b( ( float *)dp1, LOG2_SIZE, LOG2_SIZE ) ;

	dbg_p_d_data_f ( "after iwhm", ( float *)dp1, LOG2_SIZE ) ;

}
