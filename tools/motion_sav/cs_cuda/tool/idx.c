#include <stdio.h>
#include <stdlib.h>
#include <math.h>

loopie( int offset, int size )
{
	int i, j, k ;

	printf("offset === %d size %d\n", offset, size ) ;

	for ( i = 0 ; i < size ; i++ )
	{
		j = ( i / offset ) * ( offset << 1 ) ;
		// printf("			i = %d --> %d\n", i, j ) ;

		j = j + ( i % offset ) ;

		printf("	i = %d --> %d\n", i, j ) ;
	}
}

main( int ac, char *av[] )
{
	int i, offset, j, size ;

	if ( ac <= 1 )
	{
		printf("Usage: %s log2(size) \n", av[0] ) ;
		printf("	to calculate the idx of each iteration in fast transform\n") ;
		printf("	using WH-maxtrix\n") ;
		exit( 1 ) ;
	}

	size = atoi(( char *)av[1] ) ;
	size = pow( 2, size ) ;

	offset = 1 ;

	for ( i = 0 ; offset < size ; i++ )
	{	
		loopie ( offset, size >> 1 ) ;
		offset <<= 1 ;
	}
}
