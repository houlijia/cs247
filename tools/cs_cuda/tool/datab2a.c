#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#define SIZE 1024

#define SWAP(x)	(((x << 24 ) &0xff000000 ) | \
		(( x << 8 ) & 0xff0000 ) | \
		(( x >> 8 ) & 0xff00 ) | \
		(( x >> 24 ) & 0xff ))

usage( char *s )
{
	printf("Usage: %s [-c] -i filename [-s]\n", s) ;
	printf("	s: swap\n") ;
	printf("	c: char\n") ;
}

main( int ac, char *av[] )
{
	int doswap = 0, i, s, fd, total, size, cnt, dochar = 0 ;
	int *oip, *ip ;
	unsigned char *ocp, *cp ;
	char opt, *finname = NULL ;

#ifdef CUDA_OBS 
	i = 0x12345678 ;
	s = SWAP( i ) ;

	fprintf( stderr, "O %8.8x N %8.8x\n", i, s ) ;
#endif 

	while ((opt = getopt(ac, av, "ci:ds")) != -1)
	{
		switch (opt) {
		case 's' :
			doswap = 1 ;
			break ;

		case 'c' :
			dochar = 1 ;
			break ;

		case 'i' :
			finname = optarg ;
			break ;
		}
	}

	if ( finname == NULL  )
	{
		usage( av[0] ) ;
		exit( 1 ) ;
	}

	if ( dochar )
	{
		size = SIZE ;
		ocp = malloc ( size ) ;
	} else
	{
		size = SIZE * sizeof ( int ) ;
		oip = malloc ( size ) ;
	}

	fprintf( stderr, "%s: dochar %d finname %s swap %d\n", av[0], dochar,
		finname, doswap ) ;

	fd = open ( finname, O_RDONLY ) ;

	if ( fd < 0 )
	{
		printf("file open failed: errno %d\n", errno ) ;
		exit( 1 ) ;
	}

	fprintf( stderr, "dochar %d size %d\n", dochar, size ) ;

	total = 0 ;
	while ( 1 )
	{
		if ( dochar )
			cnt = read ( fd, ocp, size ) ; 
		else
			cnt = read ( fd, oip, size ) ; 

		if ( !cnt )
		{
			fprintf( stderr, "total %d \n", total ) ;
			exit( 0 ) ;
		}

		if ( cnt < 0 )
		{
			printf("read fail errno %d total %d \n", errno, total ) ;
			exit( 0 ) ;
		}

		if ( !dochar )
		{
			if ( cnt % sizeof( int ))
			{
				printf("read fail cnt %d total %d \n", cnt, total ) ;
				exit( 0 ) ;
			}
			cnt >>= 2 ;
		}

		total += cnt ;

		if ( dochar )
		{
			cp = ocp ;
			while ( cnt-- )
			{
				printf("%d\n", *cp++ ) ;
			}
		} else
		{
			ip = oip ;
			while ( cnt-- )
			{
				s = *ip++ ;

				if ( doswap )
					s = SWAP(s) ;

				printf("%d\n", s ) ;
			}
		}
	}
}
