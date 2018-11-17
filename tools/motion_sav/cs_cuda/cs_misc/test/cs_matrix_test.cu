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
#include "cs_matrix.h"

#define CUDA_DBG
#define BUF_SIZE_INT		10
#define BUF_SIZE_INT1		8	

int *dp1, *dp2, *dp3 ;
int *dp11, *dp22, *dp33 ;

float *hffp, *hffp1, *ffp1, *ffp2 ;

int buf1[ BUF_SIZE_INT ] ; 
float dbuf[ BUF_SIZE_INT ] ;

float fres, *fp1 ;
float hd1[] = { 0.2, 0.4, 0.6 } ;
float hd2[] = { 1.2, 1.4, 1.6 } ;

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
	int h, v, t, c, ii, jj, kk, ll, i, j, k, *dp ;
	float d, *dbp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	if (( k = cudaMalloc( &dp1, sizeof ( int ) * BUF_SIZE_INT * 3 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	if (( k = cudaMalloc( &fp1, sizeof ( float ) * BUF_SIZE_INT * 3 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	dp2 = dp1 + BUF_SIZE_INT ;
	dp3 = dp2 + BUF_SIZE_INT ;

	// test h_do_vector_add_destroy : even entries

	dp = buf1 ;

	i = 0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	i = h_do_vector_add_destroy( dp1, BUF_SIZE_INT ) ;

	printf("h_do_vector_add_destroy: %d \n", i ) ;

	// test h_do_vector_add_destroy : odd entries

	dp = buf1 ;

	i = 0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	i = h_do_vector_add_destroy( dp1, BUF_SIZE_INT - 1 ) ;

	printf("h_do_vector_add_destroy: %d \n", i ) ;

	// test h_do_dot : even entries 

	dp = buf1 ;
	i = 0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	dp = buf1 ;
	i = 100 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp2, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	i = h_do_dot( dp1, dp2, dp3, BUF_SIZE_INT ) ;

	printf("h_do_dot: %d \n", i ) ;

	// test h_do_dot : odd entries 

	dp = buf1 ;
	i = 0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	dp = buf1 ;
	i = 100 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp2, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	i = h_do_dot( dp1, dp2, dp3, BUF_SIZE_INT - 1 ) ;

	printf("h_do_dot: %d \n", i ) ;

	// test of h_do_scale_mul_vector : int

	dp = buf1 ;
	i = 0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dp++ = i++ ;

	put_d_data_i( dp1, buf1, sizeof( int ) * BUF_SIZE_INT ) ;

	h_do_scale_mul_vector ( dp1, 3, BUF_SIZE_INT ) ;

	cudaMemcpy( buf1, dp2, sizeof( int ) * BUF_SIZE_INT, cudaMemcpyDeviceToHost ) ;

	dp = buf1 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		printf("%d : %i \n", k, *dp++ ) ;

	// test of h_do_scale_mul_vector : float

	dbp = dbuf ;
	d = 0.0 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dbp++ = d++ ;

	cudaMemcpy( dp1, dbuf, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	h_do_scale_mul_vector ( ( float *)dp1, ( float )0.5, BUF_SIZE_INT, ( float *)dp2 ) ;

	dbg_p_d_data_f("after dp1", ( float *)dp1, BUF_SIZE_INT ) ;

	dbg_p_d_data_f("after dp2", ( float *)dp2, BUF_SIZE_INT ) ;

	printf("------------------------------------------------------------------------\n") ;

	// test of h_do_scale_mul_vector : float 

	// cudaMemcpy( dp1, dbuf, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;
	dbg_p_d_data_f("before dp1", ( float *)dp1, BUF_SIZE_INT ) ;

	h_do_scale_mul_vector ( ( float *)dp1, ( float )0.5, BUF_SIZE_INT, ( float *)dp1 ) ;

	dbg_p_d_data_f("after dp1", ( float *)dp1, BUF_SIZE_INT ) ;

	printf("------------------------------------------------------------------------\n") ;

	// testing h_do_dot in float

	put_d_data_f ( fp1, hd1, sizeof ( float ) * 3 ) ;
	put_d_data_f ( fp1 + 3, hd2, sizeof ( float ) * 3 ) ;
	fres = h_do_dot ( fp1, fp1 + 3, fp1 + 6, 3 ) ;
	printf("fres %f \n", fres ) ;

	dbg_p_d_data_f ("result ", fp1 + 6, 3 ) ;

	fres = h_do_vector_add_destroy ( fp1 + 6, 3 ) ;

	dbg_p_d_data_f ("result1 ", fp1 + 6, 3 ) ;
	printf("fres %f \n", fres ) ;

	// testing h_do_scale_add_vector

	dbp = dbuf ;
	d = 1.1 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dbp++ = d++ ;

	cudaMemcpy( dp1, dbuf, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	h_do_scale_add_vector ( ( float *)dp1, ( float )1.1, BUF_SIZE_INT ) ;

	cudaMemcpy( dbuf, dp1, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyDeviceToHost ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_scale_add_vector caller", ( float *)dp1, BUF_SIZE_INT ) ;
#endif    

	// testing h_do_copy_vector

	h_do_copy_vector( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_copy_vector", ( float *)dp2, BUF_SIZE_INT ) ;
#endif    

	// test h_do_abs_vector

	h_do_scale_add_vector ( ( float *)dp2, -6.0, BUF_SIZE_INT ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_abs_vector", ( float *)dp2, BUF_SIZE_INT ) ;
#endif    

	h_do_abs_vector (( float *)dp2, BUF_SIZE_INT ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_copy_vector OUT", ( float *)dp2, BUF_SIZE_INT ) ;
#endif    

	// test h_do_vector_sub_vector

	dbp = dbuf ;
	d = 1.1 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dbp++ = d++ ;

	cudaMemcpy( dp1, dbuf, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	dbp = dbuf ;
	d = 8.1 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
		*dbp++ = d++ + 0.1 * k ;

	cudaMemcpy( dp2, dbuf, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	h_do_vector_sub_vector (( float *)dp1, ( float *)dp2, ( float *)dp3, BUF_SIZE_INT ) ;

#ifdef CUDA_DBG 

	dbg_p_d_data_f("h_do_vector_sub_vector in1", ( float *)dp1, BUF_SIZE_INT ) ;
	dbg_p_d_data_f("h_do_vector_sub_vector in2", ( float *)dp2, BUF_SIZE_INT ) ;
	dbg_p_d_data_f("h_do_vector_sub_vector OUT", ( float *)dp3, BUF_SIZE_INT ) ;
#endif    

	// testing h_do_vector_add_vector

	h_do_vector_add_vector (( float *)dp2, ( float *)dp3, ( float *)dp1, BUF_SIZE_INT ) ;

#ifdef CUDA_DBG 

	dbg_p_d_data_f("h_do_vector_add_vector in1", ( float *)dp3, BUF_SIZE_INT ) ;
	dbg_p_d_data_f("h_do_vector_add_vector in2", ( float *)dp2, BUF_SIZE_INT ) ;
	dbg_p_d_data_f("h_do_vector_add_vector OUT", ( float *)dp1, BUF_SIZE_INT ) ;
#endif    

	printf("testing h_do_vector_2_norm size 10 ---------------------------------------\n") ;

	dbp = dbuf ;
	d = 1.1 ;
	for ( k = 0 ; k < BUF_SIZE_INT ; k++ )
	{
		if ( k & 1 )
			*dbp++ = d++ ;
		else
			*dbp++ = -d++ ;
	}

	cudaMemcpy( dp1, dbuf, sizeof( d ) * BUF_SIZE_INT, cudaMemcpyHostToDevice ) ;

	d = h_do_vector_2_norm( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT ) ;

	printf("h_do_vector_2_norm: d %f \n", d ) ;

	h_do_copy_vector( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT ) ;
	d = h_do_max_destroy( ( float *)dp2, BUF_SIZE_INT ) ;

	printf("h_do_max_destroy: d %f \n", d ) ;

	d = h_do_vector_inf_norm ( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT ) ;

	printf("h_do_vector_inf_norm: d %f \n", d ) ;

	printf("testing h_do_vector_2_norm size 8 ---------------------------------------\n") ;

	dbp = dbuf ;
	d = 8.1 ;
	for ( k = 0 ; k < BUF_SIZE_INT1 ; k++ )
	{
		if ( k & 1 )
			*dbp++ = d-- ;
		else
			*dbp++ = -d-- ;
	}

	cudaMemcpy( dp1, dbuf, sizeof( d ) * BUF_SIZE_INT1, cudaMemcpyHostToDevice ) ;

	d = h_do_vector_2_norm( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT1 ) ;

	printf("h_do_vector_2_norm: d %f \n", d ) ;

	h_do_copy_vector( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT1 ) ;
	d = h_do_max_destroy( ( float *)dp2, BUF_SIZE_INT1 ) ;

	printf("h_do_max_destroy: d %f \n", d ) ;

	d = h_do_vector_inf_norm ( ( float *)dp1, ( float *)dp2, BUF_SIZE_INT1 ) ;

	printf("h_do_vector_inf_norm: d %f \n", d ) ;

	printf("testing h_do_vhtc_2_hvtc ------------------------------------------------\n") ;

	h = 4 ;
	v = 3 ;
	t = 2;
	c = 3 ;

	i = h * v * t * c ;

	if (( k = cudaMalloc( &ffp1, sizeof ( float ) * i * 2 )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	ffp2 = ffp1 + i ;

	hffp1 = hffp = ( float * )malloc ( sizeof ( float ) * i ) ;
	
	fres = 1.1 ;
	ii = i ;
	while ( ii-- )
		*hffp1++ = fres++ ;
#ifdef CUDA_OBS 
	for ( ii = 0 ; ii < c ; ii++ )
	{
		for ( jj = 0 ; jj < t ; jj++ )	
		{
			for ( ll = 0 ; ll < h ; ll++ )
			{
				for ( kk = 0 ; kk < v ; kk++ )
				{
					j = ii * ( t * h * v ) + jj * h * v + kk *h + ll ;
					hffp1[ j ] = fres ;
			  		fres += 1.0 ;	
				}
			}
		}
	}
#endif 

	cudaMemcpy( ffp1, hffp, sizeof( float ) * i, cudaMemcpyHostToDevice ) ;
	printf("C --> MATLAB ..................................................\n") ;
	h_do_hvtc_2_vhtc ( ffp1, ffp2, v, h, t, c ) ; // c to matlab  ... this is correct ...
	printf("MATLAB --> C ..................................................\n") ;
	h_do_vhtc_2_hvtc ( ffp1, ffp2, v, h, t, c ) ; // matlab to c ... this is correct ...
}
