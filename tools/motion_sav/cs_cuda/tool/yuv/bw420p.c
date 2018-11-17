#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

// change the suspected u & v field to 0 ...

void fix_it ( int fin, int fout ) ;
char *buf ;
int Ysize, UVsize ;

main( int ac, char *av[] )
{
	int fin, fout ;
	int xsize ;

	if ( ac < 5 )
	{
		printf("Usage: %s yuv-file-in X Y yuv-file-out // 420\n", av[0]) ;
		exit( 1 ) ;
	}

	xsize = atoi ( av[2] ) ;
	Ysize = atoi ( av[3] ) ;

	Ysize *= xsize ;
	UVsize = Ysize / 2 ;

	buf = ( char * ) malloc ( Ysize ) ;

	if ( buf == NULL )
	{
		printf("malloc failed \n") ;
		exit ( 1 ) ;
	}

	fin = open( av[1], O_RDONLY ) ;

	if ( fin == -1 )
	{
		printf("file %s does not exist\n", av[1]) ;
		printf("Usage: %s yuv-file-in yuv-file-out\n", av[0]) ;
		exit( 1 ) ;
	}

	fout = open( av[4], O_CREAT | O_TRUNC | O_RDWR ) ;

	if ( fout == -1 )
	{
		printf("file %s open failed %d\n", av[1], errno ) ;
		printf("Usage: %s yuv-file-in yuv-file-out\n", av[0]) ;
		exit( 1 ) ;
	}

	fix_it( fin, fout ) ;

	close ( fin ) ;
	close ( fout ) ;

}

void
fix_it ( int fin, int fout )
{
	int i, j, total, osize, size ;
	char *cp ;
	off_t ck, offset = 0 ;

	total = 0 ;
	while ( 1 )
	{
		if (( i = read ( fin, buf, Ysize )) <= 0 )
		{
			printf("fix_it:Y: overall size %d\n", total ) ;
			return ;
		}

		if ( i != Ysize )
		{
			printf("fix_it:Y: failed i %d total %d\n",
				i, total ) ;
			return ;
		}

		if (( i = write ( fout, buf, Ysize )) != Ysize )
		{
			printf("fix_it:Y: failed %d %d %d\n", errno, i, fout )	;
			return ;
		}

		total += Ysize ;

		if (( i = read ( fin, buf, UVsize )) != UVsize )
		{
			printf("fix_it: fail  overall size %d\n", total ) ;
			return ;
		}

		cp = buf ;
		while ( i-- )
		   *cp++ = 0x7f ;

		if (( i = write ( fout, buf, UVsize )) != UVsize )
		{
			printf("fix_it: failed %d %d %d\n", errno, i, fout )	;
			return ;
		}

		total += UVsize ;
	}

	printf("fix_it: overall size %d\n", total ) ;
}
