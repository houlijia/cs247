#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

struct pixrank {
	int from_cnt ;
	int to_cnt ;
} ;

static struct pixrank *pr ;

pusage( char *s )
{
	fprintf( stderr, "Usage: %s -i xdim ydim zdim\n", s ) ;
}

main( int ac, char *av[] )
{
	char opt ;
	int opt_num[5], i, xdim = 0, ydim = 0, zdim = 0 ;

	while ((opt = getopt(ac, av,"i")) != -1) 
	{
		switch (opt) {
		case 'i' :
			if ( !get_nums( ac, av, optind, 3, opt_num ))
			{
				pusage( av[0] ) ;
				exit( 1 ) ;
			}
			xdim = opt_num[0] ;
			ydim = opt_num[1] ;
			zdim = opt_num[2] ;

			break ;
		}
	}

	if ( !xdim || !ydim || !zdim )
	{
		pusage( av[0] ) ;
		exit( 1 ) ;
	}

	i = xdim * ydim * zdim * sizeof ( *pr ) ;

	if (!( pr = ( struct pixrank * ) malloc ( i )))
	{
		fprintf( stderr, "%s: malloc %d failed \n", __func__, i ) ;
		exit( 3 ) ;
	}

	memset ( pr, 0, i ) ;

}

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

int 
chartoint(int c)
{
	char hex[] = "aAbBcCdDeEfF";
	int i;
	int result = 0;

	for(i = 0; result == 0 && hex[i] != '\0'; i++)
	{
		if(hex[i] == c)
		{
			result = 10 + (i / 2);
		}
	}

	return result;
}

unsigned int
htoi(const char s[])
{
	unsigned int result = 0;
	int i = 0;
	int proper = 1;
	int temp;

	//To take care of 0x and 0X added before the hex no.
	if(s[i] == '0')
	{
		++i;
		if(s[i] == 'x' || s[i] == 'X')
		{
			++i;
		}
	}

	while(proper && s[i] != '\0')
	{
		result = result * 16;
		if(s[i] >= '0' && s[i] <= '9')
		{
			result = result + (s[i] - '0');
		}
		else
		{
			temp = chartoint(s[i]);
			if(temp == 0)
			{
				proper = 0;
			}
			else
			{
				result = result + temp;
			}
		}

		++i;
	}
	//If any character is not a proper hex no. ,  return 0
	if(!proper)
	{
		result = 0;
	}

	return result;
}
