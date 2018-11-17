#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WHT_SIZE	1024
#define VEC_SIZE	( WHT_SIZE * WHT_SIZE )

int databuf[ VEC_SIZE * 2 ] ;
int *datap, *to_datap ;

#define CUDA_DBG 

void p_num_nm ( char *s, int *dp, int col, int row ) ;

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

int
make_data( int col, int row, int *framep )
{
	int vec_size, fake_data = 0, j, i, wht_size, *tp, *fp ;
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

	if ( framep == NULL )
	{
		fake_data++ ;
		framep = ( int * )malloc ( total * sizeof ( int ) ) ;
	}
	if ( !framep )
	{
		printf("%s :: framep failed \n") ;
		return ( 0 ) ;
	}

	if ( fake_data )
	{
		fp = framep ;
		for ( i = 0 ; i < total ; i++ )
			*fp++ = i + 1 ;
	}

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
	p_num_nm ("after make_data", datap, wht_size, wht_size ) ;
#endif 

	if ( fake_data )
		free( framep ) ;

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

int *
wht( int *pixp, int col, int row, int *ret_size )  
{
	int offset, cnt, j, i, *fp, total, sqr, p2cnt ;

	total = make_data ( col, row, pixp ) ;

	sqr = ( int )sqrt((double) total ) ;

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
	*ret_size = total ;
	return ( datap ) ;
}

main( int ac, char *av[] )
{
	int total, sqr, col, row ;
	int *fp ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	if ( ac != 3 )
	{
		printf("Usage: %s col row \n", av[0] ) ;
		exit( 3 ) ;
	}

	col = atoi ( av[1] ) ;
	row = atoi ( av[2] ) ;

	if (( col > WHT_SIZE ) || ( row > WHT_SIZE )) 
	{
		printf("size mismatch: row %d col %d ... double the size of WHT_SIZE\n",
			row, col ) ;
		exit( 3 ) ;
	}	

	datap = databuf ;
	to_datap = datap + VEC_SIZE ;

	fp = wht( NULL, col, row, &total ) ;

	if ( fp )
	{
		sqr = ( int )sqrt((double) total ) ;
		printf("wht return size %d sqr %d fp %p \n", total, sqr, fp ) ;
		p_num_nm("done", fp, sqr, sqr ) ;
	} else
		printf("err found\n") ;

	exit( 0 ) ;
}
