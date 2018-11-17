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
#include <ctype.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include "do_resize.h"
#include "../../proj/camera_demo/serial_wht3.h"
#include "../../cs_misc/cs_image.h"
#include "../../cs_misc/cs_helper.h"
#include "localized_Ordered_sensing.h"
#include "./main2.h"
#include "./file_io.h"
#include "Nimresize/Nimresize_initialize.h"
#include "Nimresize/Nimresize_terminate.h"

//  #define PLAN_B // ship all bits from camera to server

enum {
	CS_TIMER_TOTAL,
	CS_TIMER_RESIZE,
	CS_TIMER_LOCAL,
	CS_TIMER_CNT
} ;

#define CUDA_DBG 

struct obj_rec_param hdr ;

struct obj_param obj_param ;

int server_sock = -1 ;	// for network

unsigned char *orp = NULL, *ogp = NULL, *obp = NULL ;

int
max_log2( int i ) 
{
	int k, j ;

	k = ( int )log2(( double )i ) ;
	j = (int)pow(2.0, k ) ;

	if ( j < i )
		j = (int)pow(2.0, k + 1 ) ;

	return ( j ) ;
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

void
convert_2_rgb( unsigned char *rgbp, int col, int row )
{
	int i ;
	unsigned char *rrp, *ggp, *bbp ;

	if ( orp == NULL )
	{
		orp = ( unsigned char * )malloc ( col * row * sizeof ( unsigned char )) ;
		ogp = ( unsigned char * )malloc ( col * row * sizeof ( unsigned char )) ;
		obp = ( unsigned char * )malloc ( col * row * sizeof ( unsigned char )) ;
	}

	assert ( orp && ogp && obp ) ;

	rrp = orp ;
	ggp = ogp ;
	bbp = obp ;

	i = row * col ;
	while ( i-- )	// BGR ... now
	{
		*bbp++ = *rgbp++ ; 	
		*ggp++ = *rgbp++ ; 	
		*rrp++ = *rgbp++ ; 	
	}
}

#ifdef CUDA_OBS 
int
matrix_transpose( unsigned char *from, int col, int row )
{
	unsigned char *tp, *otp, *fp ;
	int i, j ;
	
	i = col * row ;

	otp = tp = ( unsigned char *) malloc ( i ) ;
	if ( tp == NULL )
		return ( 0 ) ;

	for ( i = 0 ; i < col ; i++ )
	{
		fp = from + i ;
		for ( j = 0 ; j < row ; j++ )
		{
			*tp++ = *fp ;
			fp += col ;
		}
	}
	
	memcpy ( from, otp, col * row ) ; 
	free ( otp ) ;
	return ( 1 ) ;
}
#endif 

// pixp ... either r,g or b in int

void
pusage( char *s )
{
	printf("Usage: %s -d col row -t ocol orow -i ifile \n", s ) ;
}

#ifdef CUDA_OBS 
unsigned char *
imresize( unsigned char *inbuf, int ocol, int owid, int ncol, int nwid )
{
}
#endif 

main( int ac, char *av[] )
{
	int opt ;
	int j, opt_num[2], size, ifid, ofid, total, col, row ;
	struct sockaddr_in img_server_addr ;
	int k, i ;
	unsigned char *ibuf, *obuf, *ccp ;
	char *server_ip, *ifile, *ofile ;
	unsigned short server_port ;
	int ocol, orow ;
	// double total_time, total_average ;
	int ii, jj, kk, *tp, *uip ;
	unsigned char *tp1, *tp2, *tp3 ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	// omp_timer_init ( CS_TIMER_CNT ) ;

	col = row = -1 ;
	ofile =ifile = NULL ;

	server_ip = NULL ;
	server_port = 0xffff ;

	ocol = orow = -1 ;

	while ((opt = getopt(ac, av, "i:dto:h:P:")) != -1)
	{
		switch (opt) {
		case 'P' :
			server_port = atoi( optarg ) ;
			break ;

		case 'h' :
			server_ip = optarg ;
			break ;

		case 'o' :
			ofile = optarg ;
			break ;

		case 'i' :
			ifile = optarg ;
			break ;

		case 't' :
			if ( !get_nums( ac, av, optind, 2, opt_num ))
			{
				pusage( av[0] ) ;
				exit( 1 ) ;
			}
			ocol = opt_num[0] ;
			orow = opt_num[1] ;

			break ;

		case 'd' :
			if ( !get_nums( ac, av, optind, 2, opt_num ))
			{
				pusage( av[0] ) ;
				exit( 1 ) ;
			}
			col = opt_num[0] ;
			row = opt_num[1] ;

			break ;
		}
	}

	printf("ifile %s ofile %s row %d col %d ocol %d orow %d\n",
		ifile, ofile, row, col, ocol, orow ) ;

	if (( ocol != 640 ) || ( col != 512 ) || ( orow != 480 ) || ( row != 512 ))
	{
		printf("%s : size err : row %d col %d ocol %d orow %d\n", __func__,
			row, col, ocol, orow ) ;
		return 0 ;
	}

	if (( server_port != 0xffff ) && ( server_ip ))
	{
		/* Create a reliable, stream socket using TCP */
		if (( server_sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
		{
			printf("socket() failed");
			exit( 1 ) ;
		}

		/* Construct the server address structure */
		memset(&img_server_addr, 0, sizeof(img_server_addr));     /* Zero out structure */
		img_server_addr.sin_family      = AF_INET;             /* Internet address family */
		img_server_addr.sin_addr.s_addr = inet_addr(server_ip);   /* Server IP address */
		img_server_addr.sin_port        = htons( server_port ); /* Server port */

		printf("family %d port %x addr %x \n",
			img_server_addr.sin_family,
			img_server_addr.sin_port,
			img_server_addr.sin_addr.s_addr ) ;

		/* Establish the connection to the echo server */
		if ( connect( server_sock, (struct sockaddr *) &img_server_addr, sizeof(img_server_addr)) < 0)
		{
			printf("connect() failed errno %d\n", errno ) ;
			exit( 1 ) ;
		}
	} else
		printf("NO REMOTE CONNECTION !!\n") ;

	ifid = open ( ifile, O_RDONLY ) ;
	ofid = open ( ofile, O_WRONLY | O_CREAT | O_TRUNC, 0777 ) ;

	size = ocol * orow * 3 ; // rgb

	printf("ofid %d ifid %d col %d row %d size %d errno %d \n", ofid, ifid, col, row,
		size, errno ) ;

	ibuf=( unsigned char *)malloc ( size ) ;
	obuf=( unsigned char *)malloc ( size ) ;

	printf("ibuf %p obuf %p \n", ibuf, obuf ) ;

	if (( ibuf == NULL ) || ( obuf == NULL ))
	{
		printf("malloc obuf/ibuf error \n") ;
		exit( 22 ) ;
	}

	i = read ( ifid, ibuf, size ) ;

	if ( i != size )
	{
		printf("read failed want %d got %d \n", size, i ) ;
		exit( 3 ) ;
	}

	close ( ifid ) ;

	// omp_timer_on( CS_TIMER_TOTAL ) ;

	// ED ... starts from here ...

	obj_param.w = htonl( col ) ;

#ifdef PLAN_B
	obj_param.size = htonl ( 0 ) ;
	obj_param.image_size = htonl( size ) ; 
#endif 

#ifndef PLAN_B
	obj_param.size = htonl (( col/4 ) * ( col/4 ) * 6 * sizeof( int )) ;
	obj_param.image_size = htonl( col * col * 3 ) ; 
	convert_2_rgb( ibuf, ocol, orow ) ;

	if ( !matrix_transpose<unsigned char> ( orp, ocol, orow ))
	{
		printf("orp transpose failure \n") ;
		exit( 0 ) ;
	}

	if ( !matrix_transpose<unsigned char> ( ogp, ocol, orow ))
	{
		printf("ogp transpose failure \n") ;
		exit( 0 ) ;
	}

	if ( !matrix_transpose<unsigned char> ( obp, ocol, orow ))
	{
		printf("obp transpose failure \n") ;
		exit( 0 ) ;
	}

	i = ocol * orow ;

	memcpy( ibuf, obp, i ) ;
	memcpy( ibuf+i, ogp, i ) ;
	memcpy( ibuf+2*i, orp, i ) ;

	// omp_timer_on ( CS_TIMER_RESIZE ) ;

	Nimresize_initialize();

	/* Invoke the entry-point functions.  You can call entry-point functions multiple times. */
	main_Nimresize( ibuf, "./resize.out", obuf );

	/* Terminate the application.  You do not need to do this more than one time. */
	Nimresize_terminate();

#ifdef CUDA_DBG 
	file_out("iclient2.out", 512 * 512 * 3, ( char *)obuf ) ;
#endif 

	// take average

	uip = ( int * )malloc( col * col * sizeof ( int )) ;

	i = col * col ;
	tp = uip ;
	tp1 = obuf + 2 * i ;	// r
	tp2 = obuf + i ;	// g
	tp3 = obuf ; 	// b
	while ( i-- )
	{
		ii = *tp1++ ;
		jj = *tp2++ ; 
		kk = *tp3++ ;
		*tp++ = ( ii + jj + kk ) / 3 ;
	}

	matrix_transpose<int>( uip, col, col ) ;

	file_out("./J.avg", col * col * sizeof( int ), ( char *)uip ) ;

	// prep J

	// already in ibuf ...

#ifdef CUDA_OBS 

	tp1 = ( int * )malloc ( col * col * 3 ) ;
	if ( tp1 == NULL )
	{
		printf("tp1 prep J malloc failed \n" ) ;
		exit ( 3 ) ; 
	}

	ucp = ( unsigned char * ) tp1 ;

	tp2 = oorp ;
	i = col * col ;
	while ( i-- )
		*ucp++ = *tp2++ ; 
		
	tp2 = oogp ;
	i = col * col ;
	while ( i-- )
		*ucp++ = *tp2++ ; 
		
	tp2 = oobp ;
	i = col * col ;
	while ( i-- )
		*ucp++ = *tp2++ ; 

	ucp = ( unsigned char * )tp1 ;

	free ( oorp ) ;
	free ( oogp ) ;
	free ( oobp ) ;
#endif 

	// omp_timer_off ( CS_TIMER_RESIZE ) ;

	// omp_timer_on ( CS_TIMER_LOCAL ) ;

	tp = lo_sensing( uip, col, row, col, row, 0, 0 ) ;

	free ( uip ) ;

	// omp_timer_off ( CS_TIMER_LOCAL ) ;

	if ( tp == NULL )
	{
		printf("lo_sensing failed \n" ) ;
		exit ( 0 ) ; 
	}

#endif 

	// send to server

	if ( server_sock > 0 )
	{
		k = sizeof ( obj_param ) ;
		j = write ( server_sock, &obj_param, k ) ;
		if ( j != k )
		{
			printf("socket write return err %d write %d\n", j, k ) ;
			exit( 3 ) ;
		}
		else
			printf("socket write return %d write %d\n", j, k ) ;

#ifndef PLAN_B
		k = ( col/4 ) * ( col/4 ) * 6 * sizeof ( int ) ;
		j = write ( server_sock, tp, k ) ;
		if ( j != k )
		{
			printf("socket write return err %d write %d\n", j, k ) ;
			exit ( 3 ) ;
		} else
			printf("socket write return %d write %d\n", j, k ) ;

		file_out("./CM.out", k, ( char *)tp ) ;

		i = htonl ( MARKER ) ;

		k = sizeof ( int ) ;
		j = write ( server_sock, &i, k ) ;
		if ( j != k )
		{
			printf("socket write return err %d write %d\n", j, k ) ;
			exit ( 3 ) ;
		} else
			printf("socket write return %d write %d\n", j, k ) ;

#endif 

#ifndef PLAN_B
		k = col * col * 3 ;
		j = write ( server_sock, obuf, k ) ;

		file_out("./J.out", k, ( char *)obuf ) ;

#else
		k = col * row * 3 ;
		j = write ( server_sock, ibuf, k ) ;
#endif 

		if ( j != k )
		{
			printf("socket write return err %d write %d\n", j, k ) ;
			exit ( 3 ) ;
		} else
			printf("socket write return %d write %d\n", j, k ) ;

		// GET THE BITS back ...
		i = read ( server_sock, &hdr, sizeof( hdr )) ;

		if ( i != sizeof( hdr ))
		{
			printf("read header failed \n") ;
			exit( 3 ) ;
		}

		i = ntohl( hdr.tag ) ;
		total = ntohl ( hdr.size ) ;
		k = ntohl ( hdr.status ) ;

		if ( i != (int)TAG_2 )
			printf("ERR tag err : %x \n", i ) ;

		printf("READ tag %x size %d status %x \n", i, total, k ) ;

		if ( k == OBJ_RECOG_ERR )
		{
			printf("obj_recog status err %x\n", k) ;
			exit( 3 ) ;
		}

		free( ibuf ) ;
		ibuf = ( unsigned char * )malloc ( total ) ;

		if ( ibuf == NULL )
		{
			printf("obj_recog ibuf malloc err\n" ) ;
			exit( 3 ) ;
		}

		k = total ;
		total = 0 ;
		ccp = ibuf ;
		while ( total != k )
		{
			i = read ( server_sock, ccp, k ) ;

			if ( i <= 0 )
			{
				printf("read i %d total %d size %d\n", i, total, k ) ;
				exit( 3 ) ;
			}

			ccp += i ;
			total += i ;

			printf("read i %d total %d \n", i, total ) ;
		}

		// 

		close ( server_sock ) ;
	}

	// omp_timer_off ( CS_TIMER_TOTAL ) ;

	// omp_timer_get ( CS_TIMER_TOTAL, &total_time, &i, &total_average ) ;
	// printf("total time %f %i %f \n", total_time, i, total_average ) ;

	// omp_timer_get ( CS_TIMER_RESIZE, &total_time, &i, &total_average ) ;
	// printf("resize time %f %i %f \n", total_time, i, total_average ) ;

	// omp_timer_get ( CS_TIMER_LOCAL, &total_time, &i, &total_average ) ;
	// printf("local time %f %i %f \n", total_time, i, total_average ) ;

	// display for debug purpose ...

	// write( ofid, tp, ( col/4 ) * ( col/4 ) * 6 * sizeof( int )) ;	// CM file
	// write( ofid, obuf, col * col * 3 ) ;	// J file
	// write( ofid, ibuf, total ) ;	// return OUT file
	close ( ofid ) ;

#ifdef CUDA_OBS 
	cs_image_init( col, row, 1 ) ; 
#endif 

#ifdef CUDA_OBS 
	// if the image is from server ... then this CUDA_DBG should be CUDA_OBS
	ibuf = ( unsigned char *)uip ;
	i = col * row ;
	ccp = ( unsigned char *)uip ;
	printf("pix data i %d col %d row %d \n", i, col, row ) ;

	while ( i-- )
		*ccp++ = *uip++ ;
#endif 

	// cs_image_show (( char *) ibuf ) ;	// ibuf is from server
	// cs_image_load ( ofile ) ;	// ibuf is from server

	sleep ( 30 ) ;

	close( server_sock ) ;
}
