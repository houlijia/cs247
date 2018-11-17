#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <ctype.h>

#ifdef CUDA_OBS 
main( int ac, char *av[] )
{
	char opt ;
	char *finname ;
	int i = -1, j = -1, dop = 0, opt_num[5] ;

	printf("Usage: %s -i fname -p -d i j\n", av[0] ) ;
	while ((opt = getopt(ac, av, "i:dp")) != -1)
	{
		switch (opt) {
		case 'p' :
			dop = 1 ;
			break ;

		case 'i' :
			finname = optarg ;
			break ;

		case 'd' :
			if ( !get_nums( ac, av, optind, 2, opt_num ))
			{
				exit( 1 ) ;
			}
			i = opt_num[0] ;
			j = opt_num[1] ;

			break ;
		}
	}
	
	printf("dop %d fname %p i %d j %d\n", dop, finname, i, j ) ;
}
#endif 

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

