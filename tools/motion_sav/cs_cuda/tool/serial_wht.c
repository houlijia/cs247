#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int *ip ;

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

p_num( char *s, int *fp, int cnt )
{
	int i ;

	printf("%s: %s fp %p cnt %i\n", __func__, s, fp, cnt ) ;

	for ( i = 0 ; i < cnt ; i++ )
		printf("%d -- %d \n", i, fp[ i ] ) ;
}

main( int ac, char *av[] )
{
	int total, offset, i, j, k ;
	int *fip, *tip, *fp, p2cnt, cnt ;

	if ( ac != 2 )
	{
		printf("Usage: %s power \n", av[0] ) ;
		exit( 3 ) ;
	}

	p2cnt = atoi ( av[1] ) ;

	total = ( int ) pow ( 2.0, ( double )p2cnt ) ;

	if (( fip = ( int * ) malloc ( sizeof ( int ) * total )) == NULL )
	{
		printf("malloc %d failed from \n", total ) ;
		exit( 4 ) ;
	}

	if (( tip = ( int * ) malloc ( sizeof ( int ) * total )) == NULL )
	{
		printf("malloc %d failed to \n", total ) ;
		exit( 4 ) ;
	}

#ifdef CUDA_OBS 
	printf("p2cnt %d total %d \n", p2cnt, total ) ; 
#endif 

	fp = fip ;
	for ( i = 0 ; i < total ; i++ )
		*fp++ = i ;

	memset ( tip, 0, sizeof ( int ) * total ) ;

#ifdef CUDA_DBG 
	printf("p2cnt %d total %d j %d \n", p2cnt, total, j ) ; 

	p_num("fp", fip, total ) ;
	p_num("tp", tip, total ) ;
#endif 

	offset = 1 ;
	cnt = 1 ;
	fp = fip ;
	while ( p2cnt > 0 )
	{
		cnt <<=1 ;
		j = total / cnt ; 

#ifdef CUDA_OBS 
		printf("loop p2cnt %d cnt %d j %d \n", p2cnt, cnt, j ) ;
#endif 
		
		for ( i = 0 ; i < j ; i++ )
			do_wht( fip + i * cnt, tip + i * cnt, cnt / 2 ) ; 

#ifdef CUDA_OBS 
		p_num("prog", tip, total ) ;
#endif 

		fp = fip ;
		fip = tip ;
		tip = fp ;

		p2cnt-- ;
	}
	p_num("done", fip, total ) ;
}
