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
#include "cs_complgrngn.h"

#define CUDA_DBG
#define BUF_SIZE_INT		10

float xerr_A[] = { 2.1, 3.1, 4.1, 5.2 } ;
float xerr_D[] = { 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2 } ;

float lambda_A[] = { 0.1, 2.1, 3.1, 4.1 } ;
float lambda_D[] = { 1.0, 2.0, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 } ;

struct beta beta ;
struct lmderr lmderr ;
struct sqrerr sqrerr ;
struct xerr xerr ;
struct lambda lambda ;

int
main( int ac, char *av[] )
{
	int k ;
	float lgr, *dp, *xA, *xD, *lA, *lD ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;



	if (( k = cudaMalloc( &xA, sizeof ( float ) * ( 24 +  100 ) )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	xD=xA + 4 ;
	lA =xD + 8 ;
    lD = lA + 4 ;	
	dp = lD + 8 ;

	put_d_data_f ( xA, xerr_A, sizeof ( float ) *4 ) ; 
	put_d_data_f ( xD, xerr_D, sizeof ( float ) *8 ) ; 

	put_d_data_f ( lA, lambda_A, sizeof ( float ) *4 ) ; 
	put_d_data_f ( lD, lambda_D, sizeof ( float ) *8 ) ; 

	beta.A = 1.1 ;
	beta.D = 2.2 ;
	// beta.final = 3.3 ;
	beta.scldA = 4.4 ;

	lmderr.A = 2.2 ;
	lmderr.D = 3.3 ;

	sqrerr.A = 4.4 ;
	sqrerr.D = 5.5 ;

	xerr.d_A = xA ;
	xerr.A_size = 4 ;
	xerr.d_D = xD ;
	xerr.D_size = 8 ;
	xerr.J = 9.9 ;

	lambda.d_A = lA ;
	lambda.A_size = 4 ;
	lambda.d_D = lD ;
	lambda.D_size = 8 ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "xerr.A", xerr.d_A, 4 ); 
	dbg_p_d_data_f ( "xerr.D ", xerr.d_D, 8 ); 
	printf("sqrerr A %f D %f \n", sqrerr.A, sqrerr.D ) ;
	printf("-------------------------------------------------------------- 1\n") ;
#endif 

	h_do_sqrerr( &sqrerr, &xerr, dp, 100 ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "xerr.A", xerr.d_A, 4 ); 
	dbg_p_d_data_f ( "xerr.D ", xerr.d_D, 8 ); 
#endif 
	printf("sqrerr A %f == 57.869995 D %f == 301.919983\n", sqrerr.A, sqrerr.D ) ;
#ifdef CUDA_OBS 
	printf("-------------------------------------------------------------- 2\n") ;
#endif 

	// test end

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "xerr.A", xerr.d_A, 4 ); 
	dbg_p_d_data_f ( "xerr.D ", xerr.d_D, 8 ); 
	dbg_p_d_data_f ( "lambda.A", lambda.d_A, 4 ); 
	dbg_p_d_data_f ( "lambda.D ", lambda.d_D, 8 ); 
	printf("lmderr A %f D %f \n", lmderr.A, lmderr.D ) ;
	printf("-------------------------------------------------------------- 3\n") ;
#endif 

	h_do_lmderr( &lmderr, &lambda, &xerr, dp, 100 ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "xerr.A", xerr.d_A, 4 ); 
	dbg_p_d_data_f ( "xerr.D ", xerr.d_D, 8 ); 
	dbg_p_d_data_f ( "lambda.A", lambda.d_A, 4 ); 
	dbg_p_d_data_f ( "lambda.D ", lambda.d_D, 8 ); 
#endif 
	printf("lmderr A %f == 40.749996D %f == 271.059998\n", lmderr.A, lmderr.D ) ;
#ifdef CUDA_OBS 
	printf("-------------------------------------------------------------- 4\n") ;
#endif 

	lgr = h_do_compLgrng( &beta, &lmderr, &sqrerr, &xerr ) ;

	printf("lgr %f == 781.135986\n", lgr ) ; 

	// dbg_p_d_data_f ( "print 4 ", fp1, 4 ); 

	// get_d_data_f ( fp1, dbuf1, 4 * sizeof ( float )) ;
	// dbg_pdata_f ( "print 4 ", dbuf1, 4 ); 
}
