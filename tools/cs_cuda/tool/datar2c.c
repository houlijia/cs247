#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#define SWAP(x)	(((x << 24 ) &0xff000000 ) | \
		(( x << 8 ) & 0xff0000 ) | \
		(( x >> 8 ) & 0xff00 ) | \
		(( x >> 24 ) & 0xff ))

pusage( char *s )
{
	printf("Usage: %s [-c] -i infile -o outfile [-s] -d m n\n", s) ;
	printf("	to change the internal data org from column first,\n") ;
	printf("	like in matlab, to row first, like in C\n") ;
	printf("	s: swap\n") ;
	printf("	c: char\n") ;
	printf("	d: x y \n") ;
}

int get_nums( int ac, char *av[], int idx, int cnt, int *np ) ;

main( int ac, char *av[] )
{
	int frame, doswap = 0, i, j, s, ofd, fd, total, iosize, unitsize,
		cnt, dochar = 0 ;
	char *cop, *cip, *cpf, *cpt ;
	char opt, *ofinname = NULL, *finname = NULL ;
	int *iip, *iop, *ipf, *ipt, m, n, opt_num[5] ;

#ifdef CUDA_OBS 
	i = 0x12345678 ;
	s = SWAP( i ) ;

	fprintf( stderr, "O %8.8x N %8.8x\n", i, s ) ;
#endif 

	while ((opt = getopt(ac, av, "co:i:ds")) != -1)
	{
		switch (opt) {
		case 'd' :
			if ( !get_nums( ac, av, optind, 2, opt_num ))
			{
				pusage( av[0] ) ;
				exit( 1 ) ;
			}
			m = opt_num[0] ;
			n = opt_num[1] ;

			break ;

		case 's' :
			doswap = 1 ;
			break ;

		case 'c' :
			dochar = 1 ;
			break ;

		case 'o' :
			ofinname = optarg ;
			break ;

		case 'i' :
			finname = optarg ;
			break ;
		}
	}

	if ( !finname || !ofinname )
	{
		pusage( av[0] ) ;
		exit( 1 ) ;
	}

	iosize = unitsize = m * n ;
	if ( dochar )
	{
		cip = ( char * )malloc ( iosize ) ; 
		cop = ( char * )malloc ( iosize ) ; 

		if ( !cip || !cop )
		{
			printf("c malloc failed errno %d\n", errno ) ;
			exit( 1 ) ;
		}
	} else
	{
		iosize = iosize * sizeof ( int ) ;
		iip = ( int * )malloc ( iosize ) ;
		iop = ( int * )malloc ( iosize ) ;

		if ( !iip || !iop )
		{
			printf("i malloc failed errno %d\n", errno ) ;
			exit( 1 ) ;
		}
	}

	fprintf( stderr, "%s: dochar %d if %s of %s swap %d -d %d %d\n",
		av[0], dochar, finname, ofinname, doswap, m, n ) ;

	fd = open ( finname, O_RDONLY ) ;
	if ( fd < 0 )
	{
		printf("file open failed: errno %d\n", errno ) ;
		exit( 1 ) ;
	}

	ofd = open ( ofinname, O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU ) ;
	if ( ofd < 0 )
	{
		printf("o file open failed: errno %d\n", errno ) ;
		exit( 1 ) ;
	}

	total = 0 ;
	frame = 0 ;
	while ( 1 )
	{
		if ( dochar )
			cnt = read ( fd, cip, iosize ) ; 
		else
			cnt = read ( fd, iip, iosize ) ; 

		if ( !cnt )
		{
			fprintf( stderr, "done ::: total %d frame %d\n", total, frame ) ;
			exit( 0 ) ;
		}

		if ( cnt != iosize  )
		{
			printf("read fail errno %d total %d size %d iosize %d frame %d\n",
				errno, total, cnt, iosize, frame ) ;
			exit( 0 ) ;
		}

		if ( dochar )
		{
			cpt= cop ;

			for ( i = 0 ; i < m ; i++ )
			{
				cpf = cip + i ;
				for ( j = 0 ; j < n ; j++ )
				{
					*cpt++ = *cpf ;
			   		cpf += m ;
				}	
			}
		} else
		{
			ipt = iop ;

			for ( i = 0 ; i < m ; i++ )
			{
				ipf = iip + i ;
				for ( j = 0 ; j < n ; j++ )
				{
					s = *ipf ;

					if ( doswap )
						s = SWAP(s) ;

					*ipt++ = s ;
			   		ipf += m ;
				}	
			}
		}

		if ( !dochar )
			cop = ( char *)iop ;

		cnt = write ( ofd, cop, iosize ) ;

		if ( cnt != iosize )
		{	
			printf("read fail errno %d total %d size %d iosize %d frame %d\n",
				errno, total, cnt, iosize, frame ) ;
			exit( 0 ) ;
		}

		total += cnt ;
		frame++ ;
	}
}

// cmd line options ... 

int
alldigit( char *s )
{
	while ( *s )
	{
		if (!( isdigit( *s )))
			return ( 0 ) ;
		s++ ;
	}   
	return ( 1 ) ;
}

int
get_nums( int ac, char *av[], int idx, int cnt, int *np )
{
	if (( idx + cnt ) <= ac )
	{
		while ( cnt-- )
		{
			if ( alldigit( av[ idx ] ))
				*np++ = atoi ( av[ idx++ ] ) ;
			else
				return ( 0 ) ;
		}   
		return ( 1 ) ;
	} else
	{   
		printf("not enough av idx %d cnt %d ac %d\n", idx, cnt, ac ) ;
		return ( 0 ) ;
	}   
}
