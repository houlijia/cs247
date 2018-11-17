#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#define SIZE 1024

int oip[ SIZE ] ;

dowrite ( int fd, int size )
{
	int cnt ;

	cnt = write ( fd, oip, size ) ; 

	if ( cnt != size )
	{
		fprintf( stderr, "%s: write %d got %d errno %d\n",
			__func__, size, cnt, errno ) ;
		exit( 0 ) ;
	}
}

main( int ac, char *av[] )
{
	int i, k, fd, total, dochar ;
	int *ip ;
	unsigned char *cp ;

	if ( ac < 2 )
	{
		printf("usage: %s outputfile [0/1=char] < inputfile\n", av[0] ) ;
		exit( 1 ) ;
	}

	if ( ac >= 3 )
		dochar = atoi( av[ 2 ] ) ;
	else
		dochar = 1 ;

	fd = open ( av[1], O_CREAT | O_TRUNC | O_RDWR, S_IRWXU ) ;

	if ( fd < 0 )
	{
		printf("file open failed: errno %d\n", errno ) ;
		exit( 1 ) ;
	}

	fprintf( stderr, "dochar %d ofile %s \n", dochar, av[1] ) ;

	cp = ( unsigned char * ) oip ;
	ip = oip ;
	total = 0 ;
	while ( 1 )
	{
		k = scanf("%d", &i ) ;

		if ( k == EOF )
		{
			k = total % SIZE ;

			if ( k )
			{
				if ( dochar )
					dowrite ( fd, k ) ;
				else
					dowrite ( fd, k * sizeof ( int )) ;
			}
				
			fprintf( stderr, "done: total %d\n", total ) ;
			exit( 0 ) ;
		}

		total++ ;

		if ( dochar )
			*cp++ = ( unsigned char ) i ;
		else
			*ip++ = i ;

		if (!( total % SIZE ))
		{
			if ( dochar )
			{
				dowrite ( fd, SIZE ) ;
				cp = ( unsigned char * ) oip ;
			} else
			{
				dowrite ( fd, SIZE * sizeof ( int )) ;
				ip = oip ;
			}
		}
	}
}
