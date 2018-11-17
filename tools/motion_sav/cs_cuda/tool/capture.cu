#include <iostream>
using namespace std;

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "cs_helper.h"
#include "cs_video_io.h"
#include "cs_config.h"

struct cs_config csc ;
int cs_config_check( struct cs_config *csp ) ;
int frame_capture_cnt ;
void take_out_y( char *fp, char *tp, int size ) ;

void
pusage( char *s )
{
	printf("Usage: %s -f configfilename.json\n", s ) ;
}

char *bufp ;

main( int ac, char *av[] )
{
	int i, yuv422size, y_size, x, y, nblk_in_x, nblk_in_y ;
	int frame_size, frame_depth, fout, opt ;
	char *configfile = NULL ;
	struct frame_list *fp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	while ((opt = getopt(ac, av, "f:")) != -1) 
	{
		printf(" opt %c \n", opt ) ;

		switch (opt) {
		case 'f' :
			configfile = optarg ;

			break ;
		}
	}

	if ( configfile == NULL )
	{
		pusage( av[0] ) ;
		return ( 1 ) ;
	}

	cs_config_init( &csc ) ;

	if ( !cs_config ( configfile, &csc ))
	{
		pusage( av[0] ) ;
		return ( 2 ) ;
	}

	cs_config_p ( &csc ) ;

	if ( !cs_config_check( &csc ))
	{
		pusage( av[0] ) ;
		return ( 1 ) ;
	}

	frame_size = csc.frame_x * csc.frame_y ;
	frame_depth = csc.z_block ;

	y_size = frame_size * frame_depth ;
	yuv422size = y_size * 2 ;

	bufp = ( char * )malloc ( yuv422size ) ;

	memset ( bufp, 0, yuv422size ) ;

	if ( bufp == NULL )
	{
		fprintf( stderr, "malloc\n" ) ;
		exit( 2 ) ;
	}

	fprintf( stderr, "x/y (%d, %d) blk x/y/z ( %d, %d, %d) out %s \n",
		csc.frame_x, csc.frame_y, csc.x_block, csc.y_block, csc.z_block, csc.foutname) ;

	fprintf( stderr, "perm %d %s\n", csc.do_permutation, csc.permdir ) ;

	frame_capture_cnt = ( csc.capture + ( csc.z_block - 1 ))/csc.z_block ;

	// image file ...

	fout = open( csc.foutname, O_CREAT | O_TRUNC | O_WRONLY, S_IRWXU ) ;

	if ( fout == -1 )
	{
		printf("file %s open failed %d\n", csc.foutname, errno ) ;
		exit( 1 ) ;
	}

	printf("%s: video_source %d video_src %s\n", __func__, csc.video_source, csc.video_src ) ;

	x = csc.frame_x + ( csc.xadd << 1 ) ;
	y = csc.frame_y + ( csc.yadd << 1 ) ;
	nblk_in_x = ( x - csc.overlap_x ) / ( csc.x_block - csc.overlap_x )  ;
	nblk_in_y = ( y - csc.overlap_y ) / ( csc.y_block - csc.overlap_y )  ;

	if ( !cs_vio_init ( csc.z_block, csc.frame_x, csc.frame_y, csc.video_src,
		nblk_in_x, nblk_in_y, csc.md_x, csc.md_y, csc.disp_th_x, csc.disp_th_y, csc.video_source,
	   csc.fps, 0, csc.ignore_edge, 1, 0.0 ))
	{
		printf("cs_vio_init failed\n") ;
		exit( 1 ) ;
	}

	cs_vio_start() ; // ok ... get ready ...

	sleep( 1 ) ;

	while ( frame_capture_cnt-- )
	{
		printf("frame_capture_cnt %d ... \n", frame_capture_cnt ) ;

		fp = cs_vio_get() ;

		for ( i = 0 ; i < frame_depth ; i++ )
			memcpy ( bufp + ( frame_size * 2 ) * i, fp->gbp + ( frame_size * i ), frame_size ) ;
		write ( fout, bufp, yuv422size ) ;

		cs_vio_put ( fp ) ;
	} 

	close ( fout ) ;
}

void
take_out_y( char *fp, char *tp, int size )
{
	tp++ ;
	while ( size-- )
	{	
		*tp++ = *fp++ ;
		tp++ ;
	}
}

int
cs_config_check( struct cs_config *csp )
{
	int err = 0 ;

	if (( csp->frame_x <= 0 ) || ( csp->frame_y <= 0 ))
	{
		fprintf( stderr, "frame size error %d %d \n",
			csp->frame_x, csp->frame_y ) ;
		err++ ;
	}

	return ( !err ) ;
}
