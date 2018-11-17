#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

main( int ac, char *av[] )
{
	int skip, f ;
	int cnt = 1024 * 1024, j, d, d1, i, *ip ;
	int total, *buf ;

	if ( ac == 2 )
		skip = 0 ;
	else 
		skip = 1 ;	

	if ( ac == 1 )
	{
		printf("Usage : %s WH_lfsr_XX_indcs.dat\n", av[0]) ;
		printf("the output is stdout, INDEX L ::: R \n") ;
		exit( 0 ) ;
	}

	total = 0 ;
	f = open ( av[1], O_RDONLY ) ;

	buf = ( int * )malloc ( cnt * sizeof ( int )) ;

	cnt = 8 ;

	fprintf( stderr, "open %s ::: f %d skip %d buf cnt %d %p \n",
		av[1], f, skip, cnt, buf ) ;

	printf("	0	0	:::	0\n") ;
	while ( i )
	{
		j = read ( f, buf, cnt * sizeof ( int )) ;

		if ( j < 0 )
		{
			fprintf( stderr, "ERROR j %d \n", j ) ;
			exit ( j ) ;
		}

		if ( j & 0x7 )
		{
			fprintf( stderr, "ERROR size j %d \n", j ) ;
			exit ( j ) ;
		}

		if ( j == 0 )
		{
			fprintf( stderr, "done total %d\n", total ) ;
			exit ( j ) ;
		}

		j /= sizeof ( int ) ;

		swap ( buf, j ) ;

		j >>= 1 ;
		ip = buf ;
		for ( i = total ; i < j + total ; i++ )
		{
			d = *ip++ ;
			d1 = *ip++ ;

			printf("	%d	%d	:::	%d\n", i + 1, d, d1 ) ;
		}
		total += j ;
	}
}	

swap( int *ip, int cnt )
{
	int a, d ;

	while ( cnt-- )
	{
		d = *ip ;

		a = (( d & 0xff ) << 24 ) |
			(( d & 0xff00 ) << 8 ) |
			(( d & 0xff0000 ) >> 8 ) |
			(( d & 0xff000000 ) >> 24 ) ;

		*ip++ = a ;
	}
}
