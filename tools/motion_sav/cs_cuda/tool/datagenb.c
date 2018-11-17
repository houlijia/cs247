#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

main( int ac, char *av[] )
{
	char c ;
	int dochar, do_random, i, j, size, loop ;
	int fout, li ;

	if ( ac < 6 )
	{
		printf("Usage: %s size loop|max_random outfile char[0/1] random[0/1]\n", av[0] ) ;
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

	fout = open( av[3], O_CREAT | O_TRUNC | O_RDWR , S_IRWXU ) ;

	if ( fout == -1 )
	{
		printf("file %s open failed %d\n", av[3], errno ) ;
		exit( 1 ) ;
	}

	dochar = atoi( av[ 4 ] ) ;
	
	do_random = atoi( av[ 5 ] ) ;

	fprintf( stderr, "option: size %d loop %d random %d char %d of %s\n",
		size, loop, do_random, dochar, av[3] ) ;

	if ( do_random )
	{
		while ( size-- )
		{
			li = (int)(random() % loop ) ;

			printf("%d\n", li ) ;

			if ( dochar )
			{
				c = li ;
				write ( fout, &c, sizeof ( char )) ;
			} else
				write ( fout, &li, sizeof ( int )) ;
		}
	} else
	{
		while ( loop-- ) 
		{
			for ( j = 0 ; j < size ; j++ )
			{
				printf("%d\n", j ) ;
				if ( dochar )
				{
					c = j ;
					write ( fout, &c, sizeof ( char )) ;
				} else
					write ( fout, &j, sizeof ( int )) ;
			}
		}
	}
}
