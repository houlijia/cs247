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
#include <sys/wait.h>

#include <arpa/inet.h>
#include <signal.h>

#include "tcp_socket/tcp_socket.h"
#include "i_recon.h"
#include "../../cs_misc/cs_image.h"
#include "../../cs_misc/cs_helper.h"
#include "tables.h"
#include "rgb2jpg.h"

enum {
	CS_TIMER_RECON,
	CS_TIMER_COUNT
} ;

struct recon_param param ;
char *ibuf ;

#define CUDA_DBG

#define NO_DISPLAY

void
pusage( char *s )
{
	printf("Usage: %s -f -p server_port -i imagefilename \n", s ) ;
}

void
kill_defunc( int i )
{
	int pid ;

	pid = wait( &i ) ;
	printf("pid %d is gone ... status %d \n", pid, i ) ;
}

struct bits_to_client hdr ;

main( int ac, char *av[] )
{
	char *cp, opt ;
	char *ofile ;
	int fmt=FORMAT_JPG, ofid, i, total, child_pid, client_sock, server_sock ;
	unsigned short port ;
	unsigned char *outp, *jpgp, *rgbp ;
	double total_time, total_average ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	port = 0xffff ;
	ofile = NULL ;

	while ((opt = getopt(ac, av, "fi:P:")) != -1)
	{
		switch (opt) {
		case 'f' :
			fmt = FORMAT_RGB ;
			break ;

		case 'i' :
			ofile = optarg ;
			break ;

		case 'P' :
			port = atoi( optarg ) ;
			break ;

		}
	}

	if (( port <= 0 ) || ( ofile == NULL )) 
	{
		pusage( av[0] );
		exit ( 3 ) ;
	}

	printf("ofile %s port %d fmt %d\n", ofile, port, fmt ) ;

	server_sock = CreateTCPServerSocket( port ) ;

	if ( signal ( SIGCLD, kill_defunc ) == SIG_ERR )
	{	
		printf("signal err %d \n", errno ) ;
	}

	while ( 1 )
	{
		client_sock = AcceptTCPConnection( server_sock ) ;

		printf("GOT client %d \n", client_sock ) ;
		child_pid = fork();
		if (!child_pid)
		{
			// child process ...
			printf("in child \n") ;

			omp_timer_init ( CS_TIMER_COUNT ) ;

#ifdef CUDA_DBG 
			ofid = open ( ofile, O_WRONLY | O_CREAT | O_TRUNC, 0777 ) ;

			if ( ofid < 0 )
				printf("err ofid %d errno %d \n", ofid, errno ) ;
#endif 

			i = read ( client_sock, &param, sizeof ( param )) ;

			if ( i != sizeof ( param ))
			{
			  printf("header read err i %d size %lu\n", i, (unsigned long)sizeof( param )) ; 
				exit( 3 ) ;
			}

			param.wht_size = ntohl( param.wht_size ) ;
			param.r = ntohl( param.r ) ;
			param.c = ntohl( param.c ) ;
			param.iter = ntohl( param.iter ) ;
			param.lambda = ntohl( param.lambda ) ;
			param.TVweight = ntohl( param.TVweight ) ;
			param.r_start = ntohl( param.r_start ) ;
			param.c_start = ntohl( param.c_start ) ;
			param.sel_idx = ntohl( param.sel_idx ) ;
			param.sel_size = ntohl( param.sel_size ) ;
			param.size = ntohl( param.size ) ;

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

			if ( !cs_image_init ( param.c, param.r, 3 ))
			{
				printf("cs_image_init err\n") ; 
				exit( 3 ) ;
			}

			ibuf = ( char * )malloc ( param.size + 1024 ) ;

			total = 0 ;
			cp = ibuf ;
			while ( total != param.size )
			{
				i = read ( client_sock, cp, param.size ) ;

				if ( i <= 0 )
				{
					printf("read i %d total %d size %d\n", i, total, param.size ) ; 
					exit( 3 ) ;
				}

#ifdef CUDA_OBS 
				if ( i != param.size )
				{
					printf("err :: read i %d size %d\n", i, param.size ) ; 
					exit( 3 ) ;
				}
#endif 

				cp += i ;
				total += i ;
				// printf("read i %d total %d \n", i, total ) ;
			}

#ifdef CUDA_DBG 
			write ( ofid, ibuf, total ) ;
			close( ofid ) ;

#endif 

			omp_timer_on ( CS_TIMER_RECON ) ;

			outp = rgbp = reconstruct (( int *) ibuf, &param, &total ) ;

			omp_timer_off ( CS_TIMER_RECON ) ;

			if ( rgbp )
			{
#ifdef NO_DISPLAY
#ifdef CUDA_OBS 
				outp = rgbp ;
				for ( i = 0 ; i < total ; i++ )
					printf("pix %i == %d \n", i, *outp++ ) ;
#endif 

				if ( fmt == FORMAT_JPG )
					outp = jpgp = rgb2jpg ( rgbp, param.c, param.r, &total ) ;

				hdr.tag = htonl( TAG_1 ) ;
				hdr.col = htonl( param.c ) ;
				hdr.row = htonl( param.r ) ;
				hdr.size = htonl( total ) ;
				hdr.format = htonl ( fmt ) ;
				hdr.t1 = hdr.t2 = hdr.t3 = 0 ;

				i = write ( client_sock, &hdr, sizeof ( hdr )) ;

				printf("send back header %d i %d \n", i, total ) ;

				if ( i != sizeof ( hdr ))
					printf("socket err : write header return %d \n", i ) ;
				else
				{
					i = write ( client_sock, outp, total ) ;
					if ( i != total )
						printf("socket err : write send %d return %d \n", total, i ) ;
					else
						printf("socket good : write send %d return %d \n", total, i ) ;
				}
#endif 
				omp_timer_get ( CS_TIMER_RECON, &total_time, &i, &total_average ) ;

				printf("reconstruction time %f %i %f \n", total_time, i, total_average ) ;	

				if ( fmt == FORMAT_JPG )
					free( jpgp ) ;

				total = 100 ;
				while ( total-- )
				{
					cs_image_show(( char *) rgbp ) ;
					sleep ( 1 ) ;
				}

				free( rgbp ) ;
			}

			exit ( 0 ) ;
		}
		close( client_sock ) ;
	}
}
