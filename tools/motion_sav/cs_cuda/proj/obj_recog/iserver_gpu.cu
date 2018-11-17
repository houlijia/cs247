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
#include <sys/times.h>

#include <arpa/inet.h>
#include <signal.h>

#include "tcp_socket/tcp_socket.h"
#include "../../cs_misc/cs_image.h"
#include "../../cs_misc/cs_recon.h"
#include "../../cs_misc/cs_buffer.h"
#include "../../cs_misc/cs_helper.h"
#include "localized_Ordered_sensing.h"

struct cs_buf_desc _cs_buf_desc ;

// #define PLAN_B

enum {
	CS_TIMER_RECON,
	CS_TIMER_COUNT
} ;

struct obj_param param ;
char *ibuf ;
char *Jbuf ;

int cmfid, jfid, ofid ;

#define MY_PATH_MAX	128

char cm_file[ MY_PATH_MAX ] ;
char J_file[ MY_PATH_MAX ] ;
char OUT_file[ MY_PATH_MAX ] ;
char S_file[ MY_PATH_MAX ] ;

struct tms tms, tms2 ;

char exec_cmd[512] ;
char SH_CMD[]="/home/ldl/mr/baotou_cs/cs/cs_cuda/proj/obj_recog/JongGOOD/aout/for_redistribution_files_only/run_aout.sh " ;

#define CUDA_DBG

#define NO_DISPLAY

void
ptime( const char *s, struct tms *from_tmsp )
{
	int i ;
	struct tms t ;

	i = times( &t ) ;

	printf("%s %d ::: tms u %ld s %ld cu %ld cs %ld\n",
		s, i,
		(long)from_tmsp->tms_utime,
		(long)from_tmsp->tms_stime,
		(long)from_tmsp->tms_cutime,
		(long)from_tmsp->tms_cstime ) ;

	printf("current u %ld s %ld cu %ld cs %ld\n",
		(long)t.tms_utime,
		(long)t.tms_stime,
		(long)t.tms_cutime,
		(long)t.tms_cstime ) ;

	printf("diff u %ld s %ld cu %ld cs %ld\n",
	       long(t.tms_utime - from_tmsp->tms_utime),
	       long(t.tms_stime - from_tmsp->tms_stime),
	       long(t.tms_cutime - from_tmsp->tms_cutime),
	       long(t.tms_cstime - from_tmsp->tms_cstime) );
}

static int tics_per_second = 1 ;

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

struct obj_rec_param hdr ;

main( int ac, char *av[] )
{
	char *cp, opt ;
	char *ofile ;
	pid_t cpid ;
	int status ;
	struct stat statbuf ;
	int j, fmt=FORMAT_JPG, ofid, i, total, child_pid, client_sock, server_sock ;
#ifndef PLAN_B
	int k;
#endif
	unsigned short port ;

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

	if ( port <= 0 ) 
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

	tics_per_second = sysconf( _SC_CLK_TCK ) ;

	printf("tics_per_second %d \n", tics_per_second ) ;

	while ( 1 )
	{
		client_sock = AcceptTCPConnection( server_sock ) ;

		printf("GOT client %d =================================================================\n",
			client_sock ) ;

		child_pid = fork();
		if (!child_pid )
		{
			// child process ...
			cpid = getpid() ;

			printf("in child %d \n", cpid ) ;

			// omp_timer_init ( CS_TIMER_COUNT ) ;

#ifdef CUDA_DBG 
			ofid = open ( ofile, O_WRONLY | O_CREAT | O_TRUNC, 0777 ) ;

			if ( ofid < 0 )
				printf("err ofid %d errno %d \n", ofid, errno ) ;
#endif 

#ifndef PLAN_B
			sprintf( cm_file, "/tmp/CM.%d.XXXXXX", cpid ) ;
#endif 
			sprintf( J_file, "/tmp/J.%d.XXXXXX", cpid ) ;
			sprintf( OUT_file, "/tmp/OUT.%d.XXXXXX", cpid ) ;
			sprintf( S_file, "/tmp/STAT.%d.XXXXXX", cpid ) ;

#ifndef PLAN_B
			cmfid = mkstemp( cm_file ) ;
			printf("files cm %s j %s o %s cmfid %d jfid %d \n", cm_file, J_file, OUT_file, cmfid, jfid ) ;
#endif 
			jfid = mkstemp( J_file ) ;
			printf("files j %s o %s jfid %d \n", J_file, OUT_file, jfid ) ;

			i = read ( client_sock, &param, sizeof ( param )) ;

			if ( i != sizeof ( param ))
			{
				printf("header read err i %d size %lu\n", i, (unsigned long)sizeof( param )) ; 
				exit( 3 ) ;
			}

			param.w = ntohl( param.w ) ;
			param.size = ntohl( param.size ) ;
			param.image_size = ntohl( param.image_size ) ;

			printf("param w %d size %d image %d --- ",
				param.w,
				param.size,
			   	param.image_size ) ;

			ibuf = ( char * )malloc ( param.size + 1024 ) ;

			total = 0 ;
			cp = ibuf ;
			while ( total != param.size )
			{
				i = read ( client_sock, cp, param.size - total ) ;

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

			printf("read %d CM\n", total ) ;

#ifndef PLAN_B
			i = write( cmfid, ibuf, total ) ;
			if ( i != total )
			{
				printf("write cm i %d total %d \n", i, total ) ; 
				exit( 3 ) ;
			}

			close( cmfid ) ;

			i = read ( client_sock, &k, sizeof ( int )) ;
			if ( i != sizeof ( int ))
			{
				printf("read i %d want 4\n", i ) ; 
				exit( 3 ) ;
			}

			if ( ntohl( k ) != MARKER )
			{
				printf("read MARKER %x want %x\n", k, MARKER ) ; 
				exit( 3 ) ;
			}

#endif 
			// read the image in ...
			Jbuf = ( char * )malloc ( param.image_size + 1024 ) ;

			total = 0 ;
			cp = Jbuf ;
			while ( total != param.image_size )
			{
				i = read ( client_sock, cp, param.image_size - total ) ;

				if ( i <= 0 )
				{
					printf("image read i %d total %d size %d\n", i, total, param.image_size ) ; 
					exit( 3 ) ;
				}

#ifdef CUDA_OBS 
				if ( i != param.image_size )
				{
					printf("err :: read i %d size %d\n", i, param.image_size ) ; 
					exit( 3 ) ;
				}
#endif 

				cp += i ;
				total += i ;
				// printf("read i %d total %d \n", i, total ) ;
			}

			printf("read %d J\n", total ) ;

			i = write( jfid, Jbuf, total ) ;

			if ( i != total )
			{
				printf("write J i %d total %d \n", i, total ) ; 
				exit( 3 ) ;
			}

			close( jfid ) ;

#ifdef CUDA_DBG 
			// write ( ofid, ibuf, total ) ;
			// write ( ofid, Jbuf, total ) ;
			// close( ofid ) ;
#endif 

#ifndef PLAN_B
			sprintf( exec_cmd, "%s /usr/local/MATLAB/R2015aSP1 %s %s %s %s",
				SH_CMD, cm_file, J_file, OUT_file, S_file ) ; 
#else
			sprintf( exec_cmd, "%s /usr/local/MATLAB/R2015aSP1 %s %s %s",
				SH_CMD, J_file, OUT_file, S_file ) ; 
#endif 

			printf("run : \"%s\"\n", exec_cmd ) ;

			j = times( &tms ) ;

			printf("i return %d \n", i ) ;

			ptime("before system", &tms ) ;

			i = system( exec_cmd ) ;
			printf("system returned : %d \n", i ) ;

			i = times( &tms2 ) ;

			printf("TIME %d i %d j %d \n", i - j, i, j ) ;

			ptime("after system", &tms ) ;


			ofid = open ( S_file, O_RDONLY ) ;
			if ( ofid < 0 )
			{
				status = OBJ_RECOG_ERR ;
				printf("read S open err errno %d \n", errno ) ;
			} else
			{
				i = read( ofid, &opt, 1 ) ;
				if ( i != 1 )
				{
					status = OBJ_RECOG_ERR ;
					printf("read S file err i %d errno %d \n", i, errno ) ;
				} else
				{
					status = ( int ) opt ;
					printf("S file status %d \n", status ) ;
				}
			}

			close ( ofid ) ;

			ofid = open ( OUT_file, O_RDONLY ) ;

			total = 0 ;
			if ( status != OBJ_RECOG_ERR )
			{
				if ( ofid < 0 )
				{
					printf("open %s failed errno %d \n", OUT_file, errno ) ;
					status = OBJ_RECOG_ERR ;
					total = 0 ;
				} else
				{
					if ( stat( OUT_file, &statbuf ) < 0 )
					{
						printf("stat %s failed errno %d \n", OUT_file, errno ) ;
						status = OBJ_RECOG_ERR ;
						total = 0 ;
						close ( ofid ) ;
					} else
					{
						total = statbuf.st_size ;
						printf("stat %s size %d \n", OUT_file, total ) ;
						if ( total < 0 )
							total = 0 ;

						Jbuf = ( char * )realloc ( Jbuf, total + 1024 ) ;
						if ( Jbuf == NULL )
						{
							printf("Jbuf alloc failed errno %d \n", errno ) ;
							status = OBJ_RECOG_ERR ;
							total = 0 ;
							close ( ofid ) ;
						} else
						{
							i = read ( ofid, Jbuf, total ) ;

							if ( i != total )
							{
								printf("Jbuf read i %d want %d %d \n", i, total, errno ) ;
								status = OBJ_RECOG_ERR ;
								total = 0 ;
								close ( ofid ) ;
							}
						}
					}
				}
			}

			hdr.tag = htonl( TAG_2 ) ;
			hdr.size = htonl( total ) ;
			hdr.status = htonl( status ) ;	// LDL

			i = write ( client_sock, &hdr, sizeof ( hdr )) ;

			printf("send back header size %d imagesize %d \n", i, total ) ;

			if ( i != sizeof ( hdr ))
				printf("socket err : write header return %d \n", i ) ;
			else
			{
				if ( total )
				{
					i = write ( client_sock, Jbuf, total ) ;	// is ibuf correct ??? LDL
					if ( i != total )
						printf("socket err : write send %d return %d \n", total, i ) ;
					else
						printf("socket good : write send %d return %d \n", total, i ) ;
				}
			}

			// save this for dbg
#ifdef CUDA_OBS 
			unlink ( cm_file ) ;
			unlink ( J_file ) ;
			unlink ( OUT_file ) ;
#endif 

			exit ( 0 ) ;
		}
		close( client_sock ) ;
	}
}
