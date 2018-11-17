#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

usage( char *s )
{
	fprintf( stderr, "Usage: %s -i filename -d keep skip\n", s) ;
}

main( int ac, char *av[] )
{
	int opt_num[2] ;
	char *line = NULL, opt, *finname = NULL ;
	FILE *fp ;
	size_t len ;
	int i, total_skip, total_keep, total, do_keep, keep, skip ;

#ifdef CUDA_OBS 
	i = 0x12345678 ;
	s = SWAP( i ) ;

	fprintf( stderr, "O %8.8x N %8.8x\n", i, s ) ;
#endif 

	while ((opt = getopt(ac, av, "ci:ds")) != -1)
	{
		switch (opt) {
		case 'd' :
			if ( !get_nums( ac, av, optind, 2, opt_num ))
			{
				usage( av[0] ) ;
				exit( 1 ) ;
			}
			keep = opt_num[0] ;
			skip = opt_num[1] ;

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

	fprintf( stderr, "%s: finname %s keep %d skip %d\n", av[0],
		finname, keep, skip ) ;

	fp = fopen ( finname, "r" ) ;

	if ( !fp )
	{
		fprintf( stderr, "file open failed: errno %d\n", errno ) ;
		exit( 1 ) ;
	}

	total = 0 ;
	total_skip = 0 ;
	total_keep = 0 ;
	i = keep ;
	do_keep = 1 ;
	while (( getline( &line, &len, fp )) != -1 )
	{
		if ( do_keep )
		{
			printf("%s",line ) ;
			i-- ;
			total_keep++ ;

			if (!i )
			{
				do_keep = 0 ;
				i = skip ;
			}
		} else
		{
			i-- ;
			total_skip++ ;
			if (!i )
			{
				do_keep = 1 ;
				i = keep ;
			}
		}
		total++ ;
	}

	fprintf( stderr, "total %d keep %d skip %d \n",
		total, total_keep, total_skip ) ;
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
		fprintf( stderr, "not enough av idx %d cnt %d ac %d\n", idx, cnt, ac ) ;
		return ( 0 ) ;
	}   
}
