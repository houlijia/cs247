#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <errno.h>

void makedata( int *dp, int size ) ;
void printdata( char *s, int *dp, int size ) ;
void makematrix( int *mp, int size, int msize ) ;
void printmatrix ( char *s, int *mp, int size ) ;
int *measuredata( int *xp, int *mp, int size ) ;

// #define CUDA_DBG 

#define BIGDIM	15

#define BIGBUF	( 32768 )

int bigbuf[ BIGBUF * BIGBUF ] ;


main( int ac, char *av[] )
{
	int dim, size, sizem ;
	int *dp2, *dp1, *dpm ;

	setbuf( stdout, NULL ) ;

	// printf("bigdim %d bigbuf %d \n", BIGDIM, BIGBUF ) ;

	if ( ac <= 1 )
	{
		printf("Usage: %s log2(size)[max 15]\n", av[0] ) ;
		exit( 1 ) ;
	}

	dim = atoi(( char *)av[1] ) ;

	if ( dim > BIGDIM )
	{	
		printf("Usage: %s log2(size)[max 15]\n", av[0] ) ;
		exit( 1 ) ;
	}

	size = pow( 2, dim ) ;

	sizem = size * size ;

	dp1 = ( int * )malloc( sizeof ( int ) * size ) ;

	if ( dp1 == NULL )
	{
		printf("malloc failed %d\n", errno ) ;
		exit( 1 ) ;
	}

#ifdef CUDA_OBS 
	dpm = ( int * )malloc( sizeof ( int ) * sizem ) ;

	if ( dpm == NULL )
	{
		printf("malloc matrix failed %d\n", errno ) ;
		exit( 1 ) ;
	}
#endif 
	dpm = bigbuf ;

#ifdef CUDA_DBG 
	printf("dp1 %p dpm %p dim %d size %d sizem %d\n", dp1, dpm, dim, size, sizem ) ;
#endif 

	makematrix( dpm, dim, size ) ;

#ifdef CUDA_DBG
	printmatrix( "init", dpm, size ) ;
#endif 

	makedata( dp1, size ) ;
	// printdata( "init", dp1, size ) ;

	dp2 = measuredata( dp1, dpm, size ) ;
	printdata( "done", dp2, size ) ;
}

int *
measuredata( int *xp, int *mp, int size )
{
	int k, i, j, *oxp, *otp, *tp ;

#ifdef CUDA_DBG 
	printf("measuredata: xp %p mp %p size %d\n", xp, mp, size ) ;
#endif 

	otp = tp = ( int * )malloc ( sizeof ( int ) * size ) ;

	if ( otp == NULL )
	{
		printf("measuredata: malloc failed %d\n", errno ) ;
		return ( NULL ) ;
	}

	oxp = xp ;

	for ( i = 0 ; i < size ; i++ )
	{
		xp = oxp ;
		k = 0 ;
		for ( j = 0 ; j < size ; j++ )
			k += ( *mp++ ) * ( *xp++ ) ;

		*tp++ = k ;
	}

	return ( otp ) ;
}

void
printmatrix ( char *s, int *mp, int size )
{
	int i, j ;

	printf("printmatrix: %s\n", s ) ;
	for ( i = 0 ; i < size ; i++ )
	{
		printf("row: %d \n", i ) ;
		for ( j = 0 ; j < size ; j++ )
			printf("	%d", *mp++ ) ;
		printf("\n") ;
	}
}

void
makematrix( int *mp, int size, int msize )
{
	int t, i, j ;
	int *omp, *fp, *tpl, *tpr, *tp ;

#ifdef CUDA_DBG 
	printf("makematrix: mp %p size %d msize %d \n", mp, size, msize ) ;
#endif 

	if ( size != 1 )
		makematrix( mp, size - 1, msize ) ;
	else
	{
		omp = mp ;
		*mp++ = 1 ;
		*mp = 1 ;
		mp = omp + msize ;
		*mp++ = 1 ;
		*mp = -1 ;

		return ;
	}

	size = pow( 2, ( size - 1 )) ;

#ifdef CUDA_DBG 
	printf("makematrix-1: mp %p size %d msize %d \n", mp, size, msize ) ;
#endif 

    for ( i = 0 ; i < size ; i++ )
	{
		fp = mp + i * msize ;
		tp = fp + size ;

		tpl = fp + size * msize ;
		tpr = tpl + size ;

		for ( j = 0 ; j < size ; j++ )
		{
			t = *fp++ ;
			*tp++ = t ;
			*tpl++ = t ;
			*tpr++ = -t ;
		}
	}
}

void
makedata( int *dp, int size )
{
	int cnt, k, i ;
	
	cnt = 0 ;
	while ( size-- )
	{
		if ( k != EOF )
			k = scanf("%d", &i ) ;

		if ( k == EOF )
			*dp++ = 0 ;
		else
		{
			cnt++ ;
			*dp++ = i ;
		}
	}
	// printf("makedta: data cnt %d\n", cnt ) ;
}	

void
printdata( char *s, int *dp, int size )
{
	int i ;

	// printf("printdata: %s\n", s ) ;
	for ( i = 0 ; i < size ; )
	{
		printf("%d -- %d\n", i, *dp++ ) ;
		i++ ;
	}
}
