#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define BUFFER_SIZE 	(19 * 1000 * 1000 )// 10000
#define NUM_MEAS		10000
#define CODE_SIZE		200
#define NUM_BINS		30000
#define MAX_LONG_LONG	20

char buffer[ BUFFER_SIZE ] ;
char pbuf[ CODE_SIZE ] ;
long long dbuf[ MAX_LONG_LONG ] ; // to handle double in coding

int bins[ NUM_MEAS ] ;	// measureament index
float meas[ NUM_MEAS ] ;

void gettype ( char **cp, int *cetype ) ;
void getint ( char **cp, int *cetype ) ;
void getsint ( char **cp, int *cetype ) ;
void getslonglong ( char **cp, long long *cetype ) ;
void getcode( char *s, char **cp, char *code, int *size ) ;
int ck_bin ( int *bp, int size, int max ) ;

main( int ac, char **av )
{
	char *nlp, *olp, *lp ;
	int *ip, kk, t, ii, jj, j, i, k, fid ;
	long long ll ;
	int ceid ;
	float ft ;
	int nbins, n_noclip, len_b, len_s ;

	if ( ac != 2 )
	{
		printf("Usage: %s csv_filename\n", av[0]) ;
		exit(3) ;
	}	   

	if (( fid = open ( av[1],  O_RDONLY )) < 0 )
	{
		printf("open %s failed %d \n", av[1], errno ) ;
		exit(1) ;
	}	   

	printf("fid %d \n", fid ) ;

	if (( i = read ( fid, buffer, BUFFER_SIZE )) < 0 )
	{
		printf("read %d failed \n", i ) ;
		exit(1) ;
	}	   

	printf("read %d buffer %p \n", i, buffer ) ;

	lp = buffer ;

	for ( j = 0 ; j <= 100000 ; j++ )
	{
		gettype ( &lp, &t ) ;

		// printf("buffer %p type %d \n", lp, t ) ;
	
		getint( &lp, &i ) ;

		// printf("buffer %p len %d \n", lp, i ) ;


		printf("---\ntype %d ::: buffer %p len %d \n", t, lp, i ) ;

		switch ( t ) {
		case 0 :
			ii = i ;

			k = 1 ;
			while ( ii )
			{
				getint ( &lp, &jj ) ;
				strncpy ( pbuf, lp, (size_t)jj ) ;
				pbuf[jj] = 0 ;
				lp += jj ;

				ii -= jj ;
				ii-- ;
				printf("type %d ::: len %2d : \"%s\" %d\n", k, jj, pbuf, ii ) ;
				k++ ;
			}
			break ;

		case 1 :
			olp = lp + i ; 

			i = 0 ;

			while ( lp < olp )
			{
				// getint ( &lp, &jj ) ;

				switch ( i ) {
				case 0 :
					getsint( &lp, &jj ) ;
					printf("n_frames : %d \n", jj ) ;
					break ;

				case 1 :
					getint( &lp, &jj ) ;
					printf("start frame : %d \n", jj ) ;
					break ;
				case 2 :
					getint( &lp, &jj ) ;
					printf("random seed : %d \n", jj ) ;
					break ;
				case 3 :
					getint( &lp, &jj ) ;
					printf("process color : %d \n", jj ) ;
					break ;
				case 4 :
					getint( &lp, &jj ) ;
					printf("wnd_type : %d \n", jj ) ;
					break ;
				case 5 :
					getint( &lp, &jj ) ;
					printf("qntzr_outrange_action : %d \n", jj ) ;
					break ;
				case 6 :
					getint( &lp, &jj ) ;
					printf("lossless_coder : %d \n", jj ) ;
					break ;
				case 7 :
					getint( &lp, &jj ) ;
					printf("conv_mode : %d \n", jj ) ;
					break ;
				case 8 :
					getint( &lp, &jj ) ;
					printf("leng(conv_rng) : %d \n", k, jj ) ;
					break ;
				case 9 :
					getint( &lp, &jj ) ;
					printf("random.rpt_spatial: %d\n", jj ) ;
					break ;
				case 10 :
					getint( &lp, &jj ) ;
					printf("random.rpt_temporal: %d\n", jj ) ;
					break ;
				case 11 :
						printf("conv_rng :") ;
						for ( k = 0 ; k < 3 ; k++ )
						{
							getint( &lp, &jj ) ;
							printf(" %d", jj ) ;
						}
						printf("\n") ;
						break ;

				case 12 :
				case 13 :
				case 14 :
				case 15 :
				case 16 :
				case 17 :
					switch ( i ) {
					case 12 :
						for ( k = 0 ; k < 3 ; k++ )
						{
							for ( kk = 0 ; kk < 6 ; kk++ )
								getint( &lp, &bins[ kk * 3 + k ]) ;
						}
						ip = bins ;
						printf("blk_size :") ;
						break ;
					case 13 :
						printf("blk_ovrlp :") ;
						break ;
					case 14 :
						printf("zero_ext_b :") ;
						break ;
					case 15 :
						printf("zero_ext_f :") ;
						break ;
					case 16 :
						printf("wrap_ext :") ;
						break ;
					case 17 :
						printf("blk_pre_diff :") ;
						break ;
					}

					for ( k = 0 ; k < 3 ; k++ )
						printf(" %d", *ip++ ) ;

					printf("\n") ;
					break ;

				case 18 :
					getint( &lp, &jj ) ;
					printf("case_no : %d\n", jj ) ;
					getint( &lp, &jj ) ;
					printf("n_cases : %d\n", jj ) ;
					break ;

				case 19 :

					k = 8 ;
					for ( k = 0 ; k < 8 ; k++ )
						getslonglong ( &lp, &dbuf[k] ) ;

					for ( k = 0 ; k < 4 ; k++ )
					{
						switch ( k ) {
						case 0 :
							printf("msrmnt_input_ratio : ") ;
							break ;
						case 1 :
							printf("qntzr_wdth_mltplr : ") ;
							break ;
						case 2 :
							printf("qntzr_ampl_stddev : ") ;
							break ;
						case 3 :
							printf("lossless_code_AC_gaus_thrsh : ") ;
						}

						ft = (float) ( dbuf[k] * pow (( double )2, dbuf[k+4] )) ;
						printf("%f f[] %lld e[] %lld \n", ft, dbuf[k], dbuf[k+4] ) ;
					}
					break ;

				case 20:
					getcode( "msrmnt_mtrx", &lp, pbuf, &k ) ;
					printf("code length %d \n", k ) ;
					break ;

				default :

					printf("ERR: unknown %d \n", jj ) ;
				}
				i++ ;
			}
			printf("\n") ;
			break ;

		case 2 :
			olp = lp + i ; 

			i = 0 ;

			while ( lp < olp )
			{
				getint ( &lp, &jj ) ;

				switch ( i ) {
				case 0 :
					printf("UV_present: %d \n", jj ) ;
					k = jj ;
					break ;
				case 1 :
					printf("precision %d \n", jj ) ;
					break ;
				case 2 :
					printf("n_frames %d \n", jj ) ;
					break ;
				case 3 :
					printf("width %d \n", jj ) ;
					break ;
				case 4 :
					printf("height %d \n", jj ) ;
					break ;
				case 5 :
					printf("seg_start_frame %d \n", jj ) ;
					break ;
				case 6 :
					printf("seg_n_frames %d \n", jj ) ;
					break ;
				case 7 :
					printf("fps %d \n", jj ) ;
					break ;
				case 8 :
					printf("UV_present %d ::: uv_ratio %d", k, jj ) ;
					break ;
				case 9 :
					printf(" %d", jj ) ;
					break ;
				case 10 :
					printf(" %d\n", jj ) ;
					break ;
				default :
					printf("ERR: unknown %d \n", jj ) ;
				}
				i++ ;
			}
			printf("\n") ;
			break ;

		case 3 :

			printf("%d :: ----------------------------------------------------------------\n", j ) ;

			olp = lp + i ; 

			i = 0 ;

			while ( lp < olp )
			{
				getint ( &lp, &jj ) ;
				if ( !i )
					printf("n_blk: ") ;
				else if ( i == 1 )
					printf(" blk: ") ;
				i++ ;
				printf("%d ", jj ) ;
			}
			printf("\n") ;
			break ;

		case 4 :
			olp = lp + i ; 

			i = 0 ;

			while ( lp < olp )
			{
				switch ( i ) {
				case 0 :
					getint ( &lp, &jj ) ;
					printf("n_rows ") ;
					printf("%d ", jj ) ;
					break ;

				case 1 :
					getint ( &lp, &jj ) ;
					printf("n_cols ") ;
					printf("%d ", jj ) ;
					break ;

				case 2 :
					getint ( &lp, &jj ) ;
					printf("is_transposed ") ;
					printf("%d ", jj ) ;
					break ;

				case 3 :
					getint ( &lp, &jj ) ;
					printf("seed ") ;
					printf("%d ", jj ) ;
					break ;

				case 4 :
					getcode( "code", &lp, pbuf, &jj ) ;
					// printf("\"%s\" : %d\n", pbuf, jj ) ;

					break ;
				case 5 :
					getint ( &lp, &jj ) ;
					printf("sqr_order ") ;
					printf("%d ", jj ) ;
					break ;

				default :
					getint ( &lp, &jj ) ;
					printf("NA ") ;
					printf("%d ", jj ) ;
				}

				i++ ;

				// printf("%p --- \n", lp ) ;
				
			}
			printf("\n") ;
			break ;

		case 5 :
			olp = lp + i ; 

			i = 0 ;
			k = 1 ;
			ii = 0 ;

			while ( lp < olp )
			{
				// printf("lp %p olp %p \n", lp, olp ) ;

				switch ( i ) {
				case 0 :
					getint ( &lp, &jj ) ;
					printf("save clipped: %d ", jj ) ;
					break ;

				default :
					getslonglong ( &lp, &dbuf[ii] ) ;
					if ( k )
					{
						printf("Sint ") ;
						k = 0 ;
					}
					// printf("lp %p %d \n", lp, jj ) ;
					printf(" %lld ", dbuf[ii] ) ;
					ii++ ;
				}

				i++ ;
			}
			printf("\n") ;

			for ( i = 0 ; i < 3 ; i++ )
			{
				switch ( i ) {
				case 0 :
					printf("q_wdth_mltplr : ") ;
					break ;
				case 1 :
					printf("q_wdth_unit : ") ;
					break ;
				case 2 :
					printf("q_ampl_mltplr : ") ;
				}

				ft = (float) ( dbuf[i] * pow (( double )2, dbuf[i+3] )) ;
				printf("%f f[] %lld e[] %lld \n", ft, dbuf[i], dbuf[i+3] ) ;
			}

			break ;

		case 6 :
			nlp = lp + i ;

			getint( &lp, &nbins ) ;
			getint( &lp, &n_noclip ) ;
			getint( &lp, &len_b ) ;
			getint( &lp, &len_s ) ;

			printf("nbin %d noclip %d lenb %d lens %d\n", nbins, n_noclip, len_b, len_s ) ;

			for ( ii = 0 ; ii < 4 ; ii++ )
				getslonglong ( &lp, &dbuf[ii] ) ;

			ft = (float) ( dbuf[0] * pow (( double )2, dbuf[2] )) ;
			printf("mean_msr f %f f[] %lld e[] %lld \n", ft, dbuf[0], dbuf[2] ) ;

			ft = (float) ( dbuf[1] * pow (( double )2, dbuf[3] )) ;
			printf("stdv_msr f %f f[] %lld e[] %lld \n", ft, dbuf[1], dbuf[3] ) ;

			// read bin ...

			for ( ii = 0 ; ii < len_b ; ii++ )
				getsint( &lp, &bins[ii ] ) ; 

			for ( ii = 0 ; ii < 3 ; ii++ )
				printf("bin %d : %d %x \n", ii, bins[ii], bins[ii] ) ;

			for ( ii = len_b - 3 ; ii < len_b ; ii++ )
				printf("bin %d : %d %x \n", ii, bins[ii], bins[ii] ) ;

			// fix up ...

			k = nbins/2 ;
			for ( ii = 0 ; ii < len_b ; ii++ )
				bins[ii] = bins[ii] + k - 1 ;

			for ( ii = 0 ; ii < 3 ; ii++ )
				printf(" --bin %d : %d %x \n", ii, bins[ii], bins[ii] ) ;

			for ( ii = len_b - 3 ; ii < len_b ; ii++ )
				printf(" --bin %d : %d %x \n", ii, bins[ii], bins[ii] ) ;

			if ( len_s )
			{
				// the saved index are the ones that should replace the elements
				// in the bins vector ... but currently we dont have this, so the
				// measurements will be 0, if the index is outside of the nbin.
				// as in ck_bin ... if there are n number of "invalid index" in the
				// bins, then there should be n number of the saved index here ...

				getsint ( &lp, &i ) ;
				printf("unexpected saved %d \n", i ) ;
				exit( 23 ) ;
			}

			if ( lp != nlp )
			{
				printf("lp %p nlp %p \n", lp, nlp ) ;
				exit( 3 ) ;
			}

			if ( k = ck_bin ( bins, len_b, nbins - 1 ))
				printf("BIN %d errors found \n", k ) ;

			break ;

		default :
			printf("%d ... unknown type \n", t ) ;
			exit(0) ;
		}
	}
}

int
ck_bin ( int *bp, int size, int max )
{
	int err, i ;

	err = 0 ;
	for ( i = 0 ; i < size ; i++ )
	{
		if (( *bp < 0 ) || ( *bp >= max ))
		{
			printf("%s: idx %d val %d \n", __func__, i, *bp ) ;
			err++ ;
		}

		bp++ ;
	}

	return ( err ) ;
}

void
gettype ( char **cp, int *cetype )
{
	char *lcp ;
	int i ;

	lcp = *cp ;
	i = *lcp++ ;
	*cetype = i & 0xff ;

	*cp = lcp ;
}
	
void
getint ( char **cp, int *cetype )
{
	char *lcp ;
	int ii = 0, i ;

	lcp = *cp ;

	while ( 1 )
	{
		i = *lcp++ ;

		// printf("ii %x i %x\n", ii, i ) ;

		ii = ii << 7 | ( i & 0x7f ) ;
		
		// printf("ii %x i %x\n", ii, i ) ;

		if (!( i & 0x80 ))
			break ;
	}

	*cetype = ii ;
	*cp = lcp ;
}

void
getsint ( char **cp, int *cetype )
{
	char *lcp ;
	int ii = 0, i ;
	int neg = 0, first = 1 ;

	lcp = *cp ;

	while ( 1 )
	{
		i = *lcp++ ;

		// printf(" getsint : ii %x i %x\n", ii, i ) ;

		if ( first )
		{
			if ( i & 0x40 )
				neg++ ;
			ii = i & 0x3f ;
			first = 0 ;
		} else
			ii = ( ii << 7 ) | ( i & 0x7f ) ;
		
		// printf(" 	getsint : ii %x i %x\n", ii, i ) ;

		if (!( i & 0x80 ))
			break ;
	}

	if ( neg )
		ii = -ii ;

	// printf("getsint: %d \n", ii ) ;

	*cetype = ii ;
	*cp = lcp ;
}

void
getslonglong ( char **cp, long long *cetype )
{
	char *lcp ;
	int neg, first = 1, i ;
	long long ii = 0 ;

	lcp = *cp ;

	neg = 0 ;
	while ( 1 )
	{
		i = *lcp++ ;

		// printf("ii %x i %x\n", ii, i ) ;

		if ( first )
		{
			if ( i & 0x40 )
				neg = 1 ;

			ii = i & 0x3f ;

			first = 0 ;
		} else
			ii = ii << 7 | ( i & 0x7f ) ;
		
		// printf("ii %lld %llx i %x\n", ii, ii, i ) ;

		if (!( i & 0x80 ))
			break ;
	}

	if ( neg )
		ii = -ii ;

	*cetype = ii ;
	*cp = lcp ;
}

void
getcode( char *s, char **cp, char *code, int *size )
{
	char *lcp ;
	int ii = 0, i ;
	int first = 1 ;

	getint( cp, &i ) ;

	// printf("%s: cp %p len %d %x\n", __func__, *cp, i, i ) ;

	*size = i ;

	lcp = *cp ;

	printf("%s \"", s ) ;
	while ( i-- )
	{
		*code++ = *lcp ;
		if ( first )
			printf("%2x-%c == ", *lcp, *lcp ) ;
		else
			printf(" %2x-%c == ", *lcp, *lcp ) ;
		lcp++ ;
	}
	*code = 0 ;
	printf("\" ") ;
	*cp = lcp ;
}
