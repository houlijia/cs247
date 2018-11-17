#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>

#include "../i_recon.h"
#include "../../../cs_misc/cs_image.h"

struct recon_param param ;
char *ibuf ;

void
pusage( char *s )
{
	printf("Usage: %s -p server_port -i imagefilename \n", s ) ;
}

main( int ac, char *av[] )
{
	char opt ;
	char *ofile ;
	int ofid, i, total, child_pid, client_sock, server_sock ;
	unsigned short port ;
	unsigned char *rgbp ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	// ofid = open ( ofile, O_WRONLY | O_CREAT | O_TRUNC, 0777 ) ;

	ofid = open ( "../wht_1280x720_1.out", O_RDONLY) ;

	if ( ofid < 0 )
	{
		printf("open failed %d \n", errno ) ;
		exit( 3 ) ;
	}

	param.wht_size = 2048 ;
	param.r = 720 ; 
	param.c = 1280 ; 
	param.iter  = 20 ;
	param.lambda = 1 ;
	param.TVweight = 20 ;
	param.r_start = 664 ;
	param.c_start = 384 ;
	param.sel_idx  = 1 ;
	param.sel_size = 460800 ;
	param.size = 5529600 ;

	printf("param wht_size %d r %d c %d iter %d lambda %f TVweight %f r_start %d c_start %d\n",
		param.wht_size,
		param.r,
		param.c,
		param.iter,
		param.lambda,
		param.TVweight,
		param.r_start,
		param.c_start ) ;

	printf("param sel_idx %d sel_size %d mea %d \n",
		param.sel_idx,
		param.sel_size,
		param.size ) ;

	if ( !cs_image_init ( param.c, param.r ))
	{
		printf("cs_image_init err\n") ; 
		exit( 3 ) ;
	}

	ibuf = ( char * )malloc ( param.size + 1024 ) ;

	i = read ( ofid, ibuf, param.size ) ;

	if ( i != param.size )
	{
		printf("read i %d size %d\n", i, param.size ) ; 
		exit( 3 ) ;
	}

#ifdef CUDA_DBG 
	write ( ofid, ibuf, i ) ;
	close( ofid ) ;

#endif 

	printf("CONSTRUCT STARTS ... \n") ;

	rgbp = reconstruct (( int *) ibuf, &param, &i ) ;

	printf("CONSTRUCT ENDS ... DISPLAY IMAGE \n") ;

	cs_image_show(( char *) rgbp ) ;

	while ( 1 )
	{
		printf("see the images ....... \n") ;
		cs_image_show(( char *) rgbp ) ;
		sleep ( 1 ) ;
	}

#ifdef CUDA_OBS 
	total = 0 ;
	while ((i = read ( client_sock, ibuf, 2014 )) > 0 )   
	{
		write ( ofid, ibuf, i ) ;
		total += i ;
	}
	printf("iserver : total %d \n", total ) ;
#endif 

	exit ( 0 ) ;
}
