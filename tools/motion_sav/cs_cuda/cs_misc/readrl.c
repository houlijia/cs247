#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>

static unsigned int *leftp, *rightp ;
static char qtmp[120] ;

static int
pv( char *s, FILE *f, unsigned int *lp, int cnt );

int main( int ac, char *av[] )
{
	int cnt, fd ;
	unsigned int size, *lp, *rp ;
	FILE *fl, *fr ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	if ( ac < 2 )
	{
		fprintf( stderr, "Usage: %s log2-cnt filename\n", av[0] ) ;
		exit( 1 ) ;
	}

	fprintf( stderr, "%s -- %s\n", av[1], av[2] ) ;

	size = atoi( av[1] ) ;

	size = ( unsigned int )pow( 2.0, ( double )size ) ;

	leftp = ( unsigned int * )malloc( sizeof ( *leftp ) * size ) ;
	rightp = ( unsigned int * )malloc( sizeof ( *leftp ) * size ) ;

	if (( rightp == NULL ) || ( leftp == NULL ))
	{
		fprintf( stderr, "malloc failed %d \n", size ) ;
		exit( 1 ) ;
	}

	lp = leftp ;
	rp = rightp ;

	fd = open ( av[2], O_RDONLY ) ;

	if ( fd < 0 )
	{
		fprintf( stderr, "open failed %d\n", errno ) ;
		exit ( 2 ) ;
	}

	sprintf( qtmp, "%s.L", av[2] ) ;
	fl = fopen ( qtmp, "w+" ) ; 

	sprintf( qtmp, "%s.R", av[2] ) ;
	fr = fopen ( qtmp, "w+") ; 

	if ( !fl || !fr )
	{
		fprintf( stderr, "open failed\n") ;
		exit( 2 ) ;
	}

	fprintf( stderr, "size %d\n", size ) ;

	*lp++ = 0 ;
	*rp++ = 0 ;

	cnt = 1 ;
	size-- ;
	while ( size-- )
	{
		if ( read( fd, lp, sizeof ( *lp )) != sizeof( *lp ))
		{	
			fprintf( stderr, "not enough cnt %d size %d\n", cnt, size ) ;
			exit( 3 ) ;
		}

		if ( read( fd, rp, sizeof ( *rp )) != sizeof( *rp ))
		{	
			fprintf( stderr, "not enough cnt %d size %d\n", cnt, size ) ;
			exit( 3 ) ;
		}

		*lp = ntohl ( *lp ) ;
		*rp = ntohl ( *rp ) ;

#ifdef CUDA_OBS 
		printf("%d : l %x -- %d == %x	%d\n", cnt, *lp, *lp, *rp, *rp ) ;
#endif 

		cnt++ ;
		rp++ ;
		lp++ ;
	}

	fprintf( stderr, "total %d \n", cnt ) ;

	pv("L", fl, leftp, cnt ) ;
	pv("R", fr, rightp, cnt ) ;

	fclose ( fl ) ;
	fclose ( fr ) ;
	
	return 0;
}

static int
pv( char *s, FILE *f, unsigned int *lp, int cnt )
{
	int i = 0 ;

	printf("%s -- \n", s ) ;
	while ( cnt-- )
	{
		fprintf( f, "%d -- %8.8x %d\n", i++, *lp, *lp ) ;
		lp++ ;
	}
	return ( 1 ) ;
}
