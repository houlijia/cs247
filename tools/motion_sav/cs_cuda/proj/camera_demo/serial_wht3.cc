#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>

#include <iostream>
using namespace std;

#include "serial_wht3.h"

// #define CUDA_DBG 

void p_num_nm ( char *s, int *dp, int col, int row ) ;
void p_num ( char *s, int *dp, int size ) ;

// cnt is half the table size
template <typename T>
void
do_wht( T *ofp, T *otp, int cnt )
{
	int i ;
	T *fp, *fp2 ; 

#ifdef CUDA_OBS 
	printf("%s: ofp %p otp %p cnt %d \n", __func__, ofp, otp, cnt ) ;
#endif 

	i = cnt ;
	fp = ofp ;
	fp2 = ofp + cnt ;
	while ( i-- )
		*otp++ = *fp++ + *fp2++ ; 

	i = cnt ;
	fp = ofp ;
	fp2 = ofp + cnt ;
	while ( i-- )
		*otp++ = *fp++ - *fp2++ ; 
}

template void do_wht<int>( int *ofp, int *otp, int cnt ) ;
template void do_wht<float>( float *ofp, float *otp, int cnt ) ;

#ifdef CUDA_OBS 
int
max_log2( int i ) 
{
	int k, j ;

	k = ( int )log2(( double )i ) ;
	j = (int)pow(2.0, k ) ;

	if ( j < i )
		j = (int)pow(2.0, k + 1 ) ;

	return ( j ) ;
}
#endif 

void
reshape ( int *tp, int *fp, int size )
{
	int from, to, i, j ;

	for ( i = 0 ; i < size ; i++ )
	{   
		for ( j = 0 ; j < size ; j++ )
		{
			from = i * size + j ;
			to = j * size + i ;

			// printf("from %d to %d \n", from, to ) ;

			tp[ to ] = fp [ from ] ;
		}
	}

	// p_num( "RESHAPE", tp, size * size ) ;
}

template<typename T>
void
p_num_nm_x ( const char *s, T *dp, int col, int row )
{	
	int i, j ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, (void*)dp, col, row ) ;

	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			cout << *dp++ << " " ;
		cout << endl ;
	}
}
template void p_num_nm_x<int> ( const char *s, int *dp, int col, int row ) ;
template void p_num_nm_x<float> ( const char *s, float *dp, int col, int row ) ;
template void p_num_nm_x<unsigned char> ( const char *s, unsigned char *dp, int col, int row ) ;

void
p_num_nm_f ( const char *s, float *dp, int col, int row )
{	
	int i, j ;
	float *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, (void *)dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%f	", *fp++ ) ;
		printf("\n") ;
	}
}

void
p_num_f( const char *s, float *fp, int cnt )
{
	int i ;

	printf("%s: %s fp %p cnt %i\n", __func__, s, (void*)fp, cnt ) ;

	for ( i = 0 ; i < cnt ; i++ )
		printf("%d -- %f \n", i, fp[ i ] ) ;
}

void
p_num_nm_uc ( const char *s, unsigned char *dp, int col, int row )
{	
	int i, j ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, dp, col, row ) ;

	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%d ", *dp++ ) ;
		printf("\n") ;
	}
}

void
p_num_nm ( const char *s, int *dp, int col, int row )
{	
	int i, j, *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, (void*)dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%d ", *fp++ ) ;
		printf("\n") ;
	}
}

void
p_num( const char *s, int *fp, int cnt )
{
	int i ;

	printf("%s: %s fp %p cnt %i\n", __func__, s, (void*)fp, cnt ) ;

	for ( i = 0 ; i < cnt ; i++ )
		printf("%d -- %d \n", i, fp[ i ] ) ;
}

template<typename T >
void
mea_un_select ( float *tp, T *fp, int total_size, int *idxp, int idx_size )
{
	int i ;

	memset( tp, 0, sizeof ( T ) * total_size ) ; 

	for ( i = 0 ; i < idx_size ; i++ )
		tp[ idxp[i] - 1 ] = *fp++ ; 
}
template void mea_un_select<float> ( float *tp, float *fp, int size, int *idxp, int idx_size ) ;
template void mea_un_select<int> ( float *tp, int *fp, int size, int *idxp, int idx_size ) ;

template<typename T>
void
mea_select ( T *tp, T *fp, int *idxp, int idx_size )
{
	int i ;

	for ( i = 0 ; i < idx_size ; i++ )
		*tp++ = fp[ idxp[ i] - 1 ] ; 
}
template void mea_select<float> ( float *tp, float *fp, int *idxp, int size ) ;
template void mea_select<int> ( int *tp, int *fp, int *idxp, int size ) ;


// At_wht_ord
// original data is in datap 
template<typename T>
T *
wht( T *to_datap, T *datap, int wht_size )  
{
	int cnt, j, i, total, p2cnt ;
	T *fp ;

	total = wht_size * wht_size ;
	p2cnt = ( int )log2(( double ) total ) ;

#ifdef CUDA_OBS 
	printf("%s : total %d p2cnt %d \n", __func__, total, p2cnt ) ;
#endif 

	cnt = 1 ;
	while ( p2cnt > 0 )
	{
		cnt <<=1 ;
		j = total / cnt ; 

#ifdef CUDA_DBG 
		printf("loop p2cnt %d cnt %d j %d \n", p2cnt, cnt, j ) ;
#endif 
		
		for ( i = 0 ; i < j ; i++ )
			do_wht<T>( datap + i * cnt, to_datap + i * cnt, cnt / 2 ) ; 

#ifdef CUDA_OBS 
		p_num("prog", to_datap, total ) ;
#endif 

		fp = datap ;
		datap = to_datap ;
		to_datap = fp ;

		p2cnt-- ;
	}

	return ( datap ) ;
}
template int * wht( int *to_datap, int *datap, int wht_size ) ;
template float * wht( float *to_datap, float *datap, int wht_size ) ;
