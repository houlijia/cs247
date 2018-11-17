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

#define CUDA_DBG 

// #define FAKE_IDX

#ifndef FAKE_IDX
#include "./sel_1m_idx.h"
#include "./sel_4m_idx.h"

int *selection_tbl[] = {
	sel_1m_idx,
	sel_4m_idx } ;

#define SEL_1M_TBL_IDX	0
#define SEL_4M_TBL_IDX	1

int sel_tbl_idx = 0 ;

#else
int sel_1m_idx[ 1024 ] ;
#endif 

int *datap, *to_datap ;

#define RGB_COMP	3

void p_num_nm ( char *s, int *dp, int col, int row ) ;
void p_num ( char *s, int *dp, int size ) ;
int *orp = NULL, *ogp = NULL, *obp = NULL ;

// cnt is half the table size
void
do_wht( int *ofp, int *otp, int cnt )
{
	int i, *fp, *fp2 ; 

#ifdef CUDA_OBS 
	printf("%s: ofp %p otp %p cnt %d \n", __func__, ofp, otp, cnt ) ;
#endif 

	i = cnt ;
	fp = ofp ;
	fp2 = ofp + cnt ;
	while ( i-- )
		*otp++ = *fp++ + *fp2++ ; 

	i = cnt ;
	fp = ofp ;
	fp2 = ofp + cnt ;
	while ( i-- )
		*otp++ = *fp++ - *fp2++ ; 
}

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

reshape ( int *tp, int *fp, int size )
{
	int from, to, i, j ;

	for ( i = 0 ; i < size ; i++ )
	{   
		for ( j = 0 ; j < size ; j++ )
		{
			from = i * size + j ;
			to = j * size + i ;

			// printf("from %d to %d \n", from, to ) ;

			tp[ to ] = fp [ from ] ;
		}
	}

	// p_num( "RESHAPE", tp, size * size ) ;
}


int
make_data( int col, int row, int *framep )
{
	int vec_size, j, i, wht_size, *tp, *fp ;
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

	if ( wht_size == 1024 )
		sel_tbl_idx = SEL_1M_TBL_IDX ;
	else if ( wht_size == 2048 )
		sel_tbl_idx = SEL_4M_TBL_IDX ;
	else
	{
		printf("%s : err wht_size %d no tbl index \n", __func__, wht_size ) ;
		return ( 0 ) ;
	}

	printf("%s : wht_size %d tbl idx %d\n", __func__, wht_size, sel_tbl_idx ) ;

	ysize = ( wht_size - row ) / 2 ;
	xsize = ( wht_size - col ) / 2 ;

	memset ( datap, 0, sizeof ( int ) * vec_size ) ;

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

void
p_num_nm ( char *s, int *dp, int col, int row )
{	
	int i, j, *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%d ", *fp++ ) ;
		printf("\n") ;
	}
}

void
p_num( char *s, int *fp, int cnt )
{
	int i ;

	printf("%s: %s fp %p cnt %i\n", __func__, s, fp, cnt ) ;

	for ( i = 0 ; i < cnt ; i++ )
		printf("%d -- %d \n", i, fp[ i ] ) ;
}

void
mea_selection ( int *tp, int *fp, int size, int sel_idx )
{
	int i, *idxp ;

	idxp = selection_tbl [ sel_idx ] ;
	for ( i = 0 ; i < size ; i++ )
		*tp++ = fp[ idxp[ i] - 1 ] ; 
}

int *
wht( int *pixp, int col, int row, int select_size )  
{
	int offset, cnt, j, i, *fp, total, sqr, p2cnt ;

	total = make_data ( col, row, pixp ) ;

	if ( !total )
		return ( NULL ) ;

	sqr = ( int )sqrt((double) total ) ;

	reshape( to_datap, datap, sqr ) ;

	fp = datap ;
	datap = to_datap ;
	to_datap = fp ;

	p2cnt = ( int )log2(( double ) total ) ;

	if ( p2cnt == 0 )
		return ( NULL ) ;

	printf("%s : total %d p2cnt %d sqr %d\n", __func__, total, p2cnt, sqr ) ;

	offset = 1 ;
	cnt = 1 ;
	while ( p2cnt > 0 )
	{
		cnt <<=1 ;
		j = total / cnt ; 

#ifdef CUDA_OBS 
		printf("loop p2cnt %d cnt %d j %d \n", p2cnt, cnt, j ) ;
#endif 
		
		for ( i = 0 ; i < j ; i++ )
			do_wht( datap + i * cnt, to_datap + i * cnt, cnt / 2 ) ; 

#ifdef CUDA_OBS 
		p_num("prog", to_datap, total ) ;
#endif 

		fp = datap ;
		datap = to_datap ;
		to_datap = fp ;

		p2cnt-- ;
	}

	p_num("RRRRR", datap, total ) ; // TTT

	mea_selection ( to_datap, datap, select_size, sel_tbl_idx ) ;  

	return ( to_datap ) ;
}

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

main( int ac, char *av[] )
{
	int sel, size, ifid, ofid, total, sqr, col, row ;
	int i, *fp, wht_size, vec_size ;
	unsigned char *ibuf, *ccp ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	if ( ac != 6 )
	{
		printf("Usage: %s col row ifile ofile sel_pert\n", av[0] ) ;
		exit( 3 ) ;
	}

	col = atoi ( av[1] ) ;
	row = atoi ( av[2] ) ;

	size = col * row * 3 ; // rgb

	ifid = open ( av[3], O_RDONLY ) ;
	ofid = open ( av[4], O_WRONLY | O_CREAT | O_TRUNC, 0777 ) ;

	sel = atoi ( av[5] ) ;

	printf("ofid %d ifid %d col %d row %d size %d errno %d sel %d\n", ofid, ifid, col, row,
		size, errno, sel ) ;

	ibuf=malloc ( size ) ;

	i = read ( ifid, ibuf, size ) ;

	if ( i != size )
	{
		printf("read failed want %d got %d \n", size, i ) ;
		exit( 3 ) ;
	}

	close ( ifid ) ;

	// fake data

#ifdef FAKE_IDX
	for ( i = 0 ; i < 1024 ; i++ )
		sel_1m_idx[i] = i + 10 ;
	
	sel_1m_idx[0] = 1 ;

	ccp = ibuf ;
	for ( i = 1 ; i < col * row + 1 ; i++ )
	{
		*ccp++ = i ;
		*ccp++ = i ;
		*ccp++ = i ;
	}
#endif 

	// fake data done

	// ED ... starts from here ...

	i = ( col > row ) ? col : row ;
	wht_size = max_log2( i ) ;

	vec_size = wht_size * wht_size ;

	sel = ( sel * col * row ) / 100 ;

	datap = malloc ( vec_size * sizeof ( int )) ;
	to_datap = malloc ( vec_size * sizeof ( int )) ;

	convert_2_rgb( ibuf, col, row ) ;

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
		}

		fp = wht( fp, col, row, sel ) ;

		if ( fp )
		{
			write ( ofid, fp, sel * sizeof ( int )) ;

			printf("wht return size %d fp %p \n", sel, fp ) ;
			// p_num("done", fp, sel ) ;
		} else
			printf("err found\n") ;
	}

	close ( ofid ) ;
}
