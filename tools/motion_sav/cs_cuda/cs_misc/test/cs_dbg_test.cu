
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
#define BUF_SIZE_INT		30

float *fp1 ;

float dbuf[] = { 0.1, 2.3, 3.0, 4.7 } ;
float dbuf1[ BUF_SIZE_INT ] ;

int *ip, ibuf[ BUF_SIZE_INT ] ;
char *cp, cbuf[ BUF_SIZE_INT ] ;

int
main( int ac, char *av[] )
{
	int i, k ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &fp1, sizeof ( float ) * BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	put_d_data_f ( fp1, dbuf, sizeof ( float ) *4 ) ; 
	dbg_p_d_data_f ( "print 4 ", fp1, 4 ); 

	get_d_data_f ( fp1, dbuf1, 4 * sizeof ( float )) ;
	dbg_pdata_f ( "print 4 ", dbuf1, 4 ); 

	for ( i = 0 ; i < BUF_SIZE_INT ; i++ )
		ibuf[i] = i ;

	ip = ( int * ) fp1; 

	put_d_data_i ( ip, ibuf, BUF_SIZE_INT * sizeof ( int )) ;

	dbg_pr_first_last( "test first_last", ip, BUF_SIZE_INT, 10 ) ;

	dbg_pr_h_first_last( "test first_last", ibuf, BUF_SIZE_INT, 10 ) ;

	for ( i = 0 ; i < BUF_SIZE_INT ; i++ )
		cbuf[i] = i ;

	cp = ( char * ) fp1 ;

	put_d_data_i ( ( int * )cp, ( int  *)cbuf, BUF_SIZE_INT * sizeof ( char )) ;
	dbg_pr_h_first_last( "test first_last", cbuf, BUF_SIZE_INT, 10 ) ;
}
