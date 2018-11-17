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

#include <sys/socket.h>
#include <arpa/inet.h>
#include "i_recon.h"
#include "tables.h"

#include "serial_wht3.h"
#include "../../cs_misc/cs_helper.h"

enum {
	CS_TIMER_TOTAL,
	CS_TIMER_WHT,
	CS_TIMER_COUNT
} ;

#define CUDA_DBG 

int sel_tbl_idx = 0 ;
int server_sock = -1 ;	// for network

int *datap, *to_datap ;

#define RGB_COMP	3

int *orp = NULL, *ogp = NULL, *obp = NULL ;

int get_nums( int ac, char *av[], int idx, int cnt, int *np ) ;

struct recon_param recon_p ;
struct bits_to_client hdr ;

void
convert_2_rgb( unsigned char *rgbp, int col, int row )
{
	int i, *rrp, *ggp, *bbp ;

	if ( orp == NULL )
	{
		orp = ( int * )malloc ( col * row * sizeof ( int )) ;
		ogp = ( int * )malloc ( col * row * sizeof ( int )) ;
		obp = ( int * )malloc ( col * row * sizeof ( int )) ;
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

// input is in framep, output is in datap 640x480 to 512x480
int
crop_data( int *datap, int col, int row, int *framep, int ocol, int orow, int *sel_tbl_idx )
{
	int vec_size, i, wht_size, *tp, *fp ;
	int row_skip, col_skip, total, ysize, xsize ;

	total = col * row ;

	i = ( col > row ) ? col : row ;

#ifdef CUDA_DBG 
	printf("%s : col %d row %d i %d total %d \n", __func__,
		col, row, i, total ) ;
#endif 

	wht_size = max_log2( i ) ;

	if (( col % 2 ) || ( row % 2 ) || ( wht_size < col ) || ( wht_size < row ))
	{
		printf("%s :: size failed col %d row %d wht_size %d \n",
			__func__, col, row, wht_size ) ;
		return ( 0 ) ;
	}

	vec_size = wht_size * wht_size ;

	switch ( wht_size ) {
	case 512 :
		*sel_tbl_idx = SEL_256K_TBL_IDX ;
		break ;

	case 1024 :
		*sel_tbl_idx = SEL_1M_TBL_IDX ;
		break ;

	case 2048 :
		*sel_tbl_idx = SEL_4M_TBL_IDX ;
		break ;
	default :
		printf("%s : err wht_size %d no tbl index \n", __func__, wht_size ) ;
		return ( 0 ) ;
	}

	printf("%s : wht_size %d tbl idx %d\n", __func__, wht_size, *sel_tbl_idx ) ;

	if ( ocol > wht_size )
		col_skip = ( ocol - wht_size ) / 2 ;
	else
		col_skip = 0 ;

	if ( orow > wht_size )
		row_skip = ( orow - wht_size ) / 2 ;
	else
		row_skip = 0 ;

	if ( ocol > wht_size )
		col_skip = ( ocol - col ) / 2 ;
	else
		col_skip = 0 ;

	if ( orow > wht_size )
		row_skip = ( orow - row ) / 2 ;
	else
		row_skip = 0 ;

	ysize = ( wht_size - row ) / 2 ;
	xsize = ( wht_size - col ) / 2 ;

	memset ( datap, 0, sizeof ( int ) * vec_size ) ;

	recon_p.wht_size = htonl( wht_size ) ;
	recon_p.r = htonl( row ) ;
	recon_p.c = htonl( col ) ;
	recon_p.r_start = htonl( ysize ) ;
	recon_p.c_start = htonl( xsize ) ;
	recon_p.sel_idx = htonl( *sel_tbl_idx ) ;

	fp = framep + row_skip * orow + col_skip ;
	tp = datap + wht_size * ysize + xsize ;

	xsize <<= 1 ;
	col_skip <<=1 ;
	
	printf("%s : wht_size %d col %d row %d fp %p tp %p vec_size %d \n", __func__,
		wht_size, col, row, fp, tp, vec_size ) ;

	for ( i = 0 ; i < row ; i++ )
	{
		memcpy ( tp, fp, sizeof( int ) * col ) ;
		tp += wht_size ;
		fp += ocol ;
	}

#ifdef CUDA_DBG 
	// p_num_nm ("after make_data", datap, wht_size, wht_size ) ;
#endif 

	return ( vec_size ) ;
}

int
make_data( int *datap, int col, int row, int *framep, int *sel_tbl_idx )
{
	int vec_size, i, wht_size, *tp, *fp ;
	int total, ysize, xsize ;

	total = col * row ;

	i = ( col > row ) ? col : row ;

#ifdef CUDA_DBG 
	printf("%s : col %d row %d i %d total %d \n", __func__,
		col, row, i, total ) ;
#endif 

	wht_size = max_log2( i ) ;

	if (( col % 2 ) || ( row % 2 ) || ( wht_size < col ) || ( wht_size < row ))
	{
		printf("%s :: size failed col %d row %d wht_size %d \n",
			__func__, col, row, wht_size ) ;
		return ( 0 ) ;
	}

	vec_size = wht_size * wht_size ;

	switch ( wht_size ) {
	case 512 :
		*sel_tbl_idx = SEL_256K_TBL_IDX ;
		break ;

	case 1024 :
		*sel_tbl_idx = SEL_1M_TBL_IDX ;
		break ;

	case 2048 :
		*sel_tbl_idx = SEL_4M_TBL_IDX ;
		break ;
	default :
		printf("%s : err wht_size %d no tbl index \n", __func__, wht_size ) ;
		return ( 0 ) ;
	}

	printf("%s : wht_size %d tbl idx %d\n", __func__, wht_size, *sel_tbl_idx ) ;

	ysize = ( wht_size - row ) / 2 ;
	xsize = ( wht_size - col ) / 2 ;

	memset ( datap, 0, sizeof ( int ) * vec_size ) ;

	recon_p.wht_size = htonl( wht_size ) ;
	recon_p.r = htonl( row ) ;
	recon_p.c = htonl( col ) ;
	recon_p.r_start = htonl( ysize ) ;
	recon_p.c_start = htonl( xsize ) ;
	recon_p.sel_idx = htonl( *sel_tbl_idx ) ;

	fp = framep ;
	tp = datap ;

	tp += wht_size * ysize + xsize ;

	xsize <<= 1 ;
	
	printf("%s : wht_size %d col %d row %d fp %p tp %p vec_size %d \n", __func__,
		wht_size, col, row, fp, tp, vec_size ) ;

	for ( i = 0 ; i < row ; i++ )
	{
		memcpy ( tp, fp, sizeof( int ) * col ) ;
		tp += wht_size ;
		fp += col ;
	}

#ifdef CUDA_DBG 
	// p_num_nm ("after make_data", datap, wht_size, wht_size ) ;
#endif 

	return ( vec_size ) ;
}

// pixp ... either r,g or b in int

int *
do_measurement( int *to_datap, int *datap, int *pixp, int col, int row, int select_size,
	int ocol, int orow )  
{
	int sel_tbl_idx ;
	int *fp, total, sqr ;

	// total = make_data ( datap, col, row, pixp, &sel_tbl_idx ) ;

	total = crop_data( datap, col, row, pixp, ocol, orow, &sel_tbl_idx ) ;

	if ( !total )
		return ( NULL ) ;

	sqr = ( int )sqrt((double) total ) ;

#ifdef CUDA_OBS 
	reshape( to_datap, datap, sqr ) ;

	fp = datap ;
	datap = to_datap ;
	to_datap = fp ;
#endif 

	fp = wht<int>( to_datap, datap, sqr ) ;

	// p_num("RRRRR", fp, total ) ; // TTT

	if ( fp != datap )
	{
		to_datap = datap ;
		datap = fp ;
	}

	printf("%s : fp %p datap %p to_datap %p \n", __func__, fp, datap, to_datap ) ;

	mea_select<int> ( to_datap, datap, selection_tbl[ sel_tbl_idx ], select_size ) ;  

	return ( to_datap ) ;
}

void
pusage( char *s )
{
	printf("Usage: %s -d col row -t ocol orow -i ifile -o ofile -p sel_pert\n", s ) ;
}

main( int ac, char *av[] )
{
	int opt ;
	int header_sent = 0, j, opt_num[2], sel, size, ifid, ofid, total, col, row ;
	struct sockaddr_in img_server_addr ;
	int k, i, wht_size, vec_size ;
	int *fp=NULL;		// Initialization to avoid warnings.
	unsigned char *ibuf, *ccp ;
	char *server_ip, *ifile, *ofile ;
	unsigned short server_port ;
	int ocol, orow ;
	double total_time, total_average ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	col = row = sel = -1 ;
	ofile =ifile = NULL ;

	server_ip = NULL ;
	server_port = 0xffff ;

	ocol = orow = 0 ;

	while ((opt = getopt(ac, av, "i:dtp:o:h:P:")) != -1)
	{
		switch (opt) {
		case 'P' :
			server_port = atoi( optarg ) ;
			break ;

		case 'h' :
			server_ip = optarg ;
			break ;

		case 'p' :
			sel = atoi( optarg ) ;
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

	if ( orow == 0 )
	{
		orow = row ;
		ocol = col ;
	}

	if (( orow < row ) || ( ocol < col ))
	{
		printf("size mismatch col %d row %d ocol %d orow %d \n", col, row, ocol, orow ) ;
		exit ( 2 ) ;
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

		/* Establish the connection to the echo server */
		if ( connect( server_sock, (struct sockaddr *) &img_server_addr, sizeof(img_server_addr)) < 0)
		{
			printf("connect() failed");
			exit( 1 ) ;
		}
	} else
		printf("NO REMOTE CONNECTION !!\n") ;

	if (( ofile == NULL ) || ( ifile == NULL ) ||
		( col <= 0 ) || ( row <= 0 ) || 
		( sel <= 0 ) || ( sel > 100 ))
	{
		pusage( av[0] );
		exit ( 3 ) ;
	}

	printf("ifile %s ofile %s row %d col %d sel %d ocol %d orow %d\n",
		ifile, ofile, row, col, sel, ocol, orow ) ;

	ifid = open ( ifile, O_RDONLY ) ;
	ofid = open ( ofile, O_WRONLY | O_CREAT | O_TRUNC, 0777 ) ;

	omp_timer_init ( CS_TIMER_COUNT ) ;

	size = ocol * orow * 3 ; // rgb

	printf("ofid %d ifid %d col %d row %d size %d errno %d sel %d\n", ofid, ifid, col, row,
		size, errno, sel ) ;

	omp_timer_on( CS_TIMER_TOTAL ) ;

	ibuf=( unsigned char *)malloc ( size ) ;

	i = read ( ifid, ibuf, size ) ;

	if ( i != size )
	{
		printf("read failed want %d got %d \n", size, i ) ;
		exit( 3 ) ;
	}

	close ( ifid ) ;

	// ED ... starts from here ...

	i = ( col > row ) ? col : row ;
	wht_size = max_log2( i ) ;

	vec_size = wht_size * wht_size ;

	sel = ( sel * col * row ) / 100 ;

	recon_p.sel_size = htonl( sel ) ;
	recon_p.size = htonl( sel * 3 * sizeof( int )) ;
	recon_p.lambda = htonl( 1 ) ;
	recon_p.TVweight = htonl ( 20 ) ;
	recon_p.iter = htonl ( 20 ) ;

	datap = ( int *)malloc ( vec_size * sizeof ( int )) ;
	to_datap = ( int *)malloc ( vec_size * sizeof ( int )) ;

	convert_2_rgb( ibuf, ocol, orow ) ;

	for ( i = 0 ; i < RGB_COMP ; i++ )
	{
		switch ( i ) {
		case 0 :
			fp = orp ;
			break ;
		case 1 :
			fp = ogp ;
			break ;
		case 2 :
			fp = obp ;
		default:
		  assert(false); // Should never get here
		}

		omp_timer_on( CS_TIMER_WHT ) ;

		fp = do_measurement( datap, to_datap, fp, col, row, sel, ocol, orow ) ;  

		omp_timer_off( CS_TIMER_WHT ) ;
		if (( server_sock > 0 ) && ( header_sent == 0 ))
		{
			header_sent++ ;
			k = sizeof ( recon_p ) ;
			j = write ( server_sock, &recon_p, k ) ;
			if ( j != k )
			{
				printf("socket write return err %d write %d\n", j, k ) ;
				exit( 3 ) ;
			}
			else
				printf("socket write return %d write %d\n", j, k ) ;
		}

		if ( fp )
		{
			write ( ofid, fp, sel * sizeof ( int )) ;

			printf("wht return size %d fp %p \n", sel, fp ) ;
			// p_num("done", fp, sel ) ;

			if ( server_sock > 0 )
			{
				j = write ( server_sock, fp, sel * sizeof ( int )) ;
				if ( j != sel * (int) sizeof ( int ))
				{
				  printf("socket write return err %d write %d\n", j, sel * (int)sizeof ( int )) ;
					exit ( 3 ) ;
				} else
				  printf("socket write return %d write %d\n", j, sel * (int) sizeof ( int )) ;

			}
		} else
			printf("err found\n") ;
	}

	omp_timer_off( CS_TIMER_TOTAL ) ;

	omp_timer_get ( CS_TIMER_TOTAL, &total_time, &i, &total_average ) ;
	printf("total time %f %i %f \n", total_time, i, total_average ) ;	

	omp_timer_get ( CS_TIMER_WHT, &total_time, &i, &total_average ) ;
	printf("wht time %f %i %f \n", total_time, i, total_average ) ;	

	// GET THE BITS back ...
	i = read ( server_sock, &hdr, sizeof( hdr )) ;

	if ( i != sizeof( hdr ))
	{
		printf("read header failed \n") ;
		exit( 3 ) ;
	}

	i = ntohl( hdr.tag ) ;
	col = ntohl ( hdr.col ) ;
	row = ntohl ( hdr.row ) ;
	total = ntohl ( hdr.format ) ;
	k = ntohl( hdr.size ) ;

	if ( i != (int)TAG_1 )
		printf("ERR tag err : %x \n", i ) ;

	printf("READ tag %x col %d row %d size %d format %x \n", i, col, row, k, total ) ;

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

	close ( ofid ) ;
	close ( server_sock ) ;
}
