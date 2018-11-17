#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>

pusage( char *s )
{
	printf("Usage: %s col row infile outfile\n", s ) ;
}

main( int ac, char *av[] )
{
	int loop, i, size, ifid, ofid, row, col ;
	char *fp, *ibuf, *obuf, *yp, *up, *vp ;

	if ( ac != 5 )
	{
		pusage( av[0] ) ;
		exit( 3 ) ;
	}

	col = atoi( av[1] ) ;
	row = atoi( av[2] ) ;

	size = col * row * 2 ; // UYVY

	ifid = open ( av[3], O_RDONLY ) ;
	ofid = open ( av[4], O_WRONLY | O_CREAT, 0777 ) ;

	printf("ofid %d ifid %d col %d row %d size %d errno %d\n", ofid, ifid, col, row, size, errno ) ;

	ibuf=malloc ( size ) ;
	obuf=malloc ( size ) ;

	i = read ( ifid, ibuf, size ) ;

	// printf("got %d ... size %d \n", i, size ) ;

	printf("obuf %p ibuf %p \n", obuf, ibuf ) ;

	close ( ifid ) ;

	if ( i != size )
	{
		printf("read failed want %d got %d \n", size, i ) ;
		exit( 3 ) ;
	}

	yp = obuf ;
	up = yp + ( size / 2 ) ;
	vp = up + ( size / 4 ) ;

	fp = ibuf ;

	loop = size / sizeof ( int ) ;

	printf("yp %p up %p vp %p loop ... %d \n", yp, up, vp, loop ) ;

	for ( i = 0 ; i < loop ; i++ )
	{
		*up++ = *fp++ ; 
		*yp++ = *fp++ ; 
		*vp++ = *fp++ ; 
		*yp++ = *fp++ ; 
	}

	i = write ( ofid, obuf, size ) ;

	if ( i != size )
	{
		printf("write failed want %d got %d errno %d\n", size, i, errno ) ;
		exit( 3 ) ;
	}
	
	close( ofid ) ;

	printf("DONE ... \n") ;
}
