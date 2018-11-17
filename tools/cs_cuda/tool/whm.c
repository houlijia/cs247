#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void makedata( int *dp, int size ) ;
void printdata( char *s, int *dp, int size ) ;
void loopdata( int *fp, int entry_per_block ) ;

// #define CUDA_DBG 

main( int ac, char *av[] )
{
	int i, size ;
	int *dp1 ;

	if ( ac <= 1 )
	{
		printf("Usage: %s log2(size) \n", av[0] ) ;
		exit( 1 ) ;
	}

	size = atoi(( char *)av[1] ) ;
	size = pow( 2, size ) ;

	dp1 = ( int * )malloc( sizeof ( int ) * size ) ;

	makedata( dp1, size ) ;
	// printdata( "init", dp1, size ) ;

	loopdata( dp1, size ) ;
	printdata( "done", dp1, size ) ;
}

void
makedata( int *dp, int size )
{
	int cnt, k, i ;
	
	cnt = 0 ;
	k = 0 ;
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
	// printf("makedata: data cnt %d\n", cnt ) ;
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

void
loopdata( int *fp, int entry_per_block )
{
	int i ;
#ifdef CUDA_DBG 
	int *origfp ;
#endif 
	int *fp2 ;
	register int t, tt ;

	i = entry_per_block >> 1 ;

	if ( entry_per_block != 2 )
	{
		loopdata( fp, i ) ; 
		loopdata( fp + i, i ) ; 
	}

#ifdef CUDA_DBG 
	origfp = fp ;
#endif 

	fp2 = fp + i ; 

	while ( i-- )
	{
		t = *fp ;
		tt = *fp2 ;
		*fp++ = t + tt ;
		*fp2++ = t - tt ;
	}

#ifdef CUDA_DBG 
	printdata( "loop", origfp, entry_per_block ) ;
#endif 
}
