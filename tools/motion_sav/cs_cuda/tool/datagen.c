#include <stdio.h>
#include <stdlib.h>

main( int ac, char *av[] )
{
	int do_random, i, j, size, loop ;
	long fout, li ;

	if ( ac <= 3 )
	{
		printf("Usage: %s size loop|max_random 0/1[1=random]\n", av[0] ) ;
		printf("	loop: number of size output in the case of the "
				"sequencial number\n") ;
		printf("	max_random: the max value of the output\n") ;
		exit( 1 ) ;
	}

	size = atoi( av[ 1 ] ) ;

	if ( size <= 0 )
	{
		printf("Usage: %s size loop \n", av[0] ) ;
		exit( 1 ) ;
	}
	
	loop = atoi( av[ 2 ] ) ;

	if ( loop <= 0 )
	{
		printf("Usage: %s size loop \n", av[0] ) ;
		exit( 1 ) ;
	}
	
	if ( ac >= 4 )
		do_random = atoi( av[ 3 ] ) ;
	else
		do_random = 1 ;

	if ( do_random )
	{
		while ( size-- )
		{
			printf("%d\n", (int)(random() % loop )) ;
		}
	} else
	{
		while ( loop-- ) 
		{
			for ( j = 0 ; j < size ; j++ )
			{
				printf("%d\n", j ) ;
			}
		}
	}
}
