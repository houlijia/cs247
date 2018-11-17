#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_buffer.h"
#include "cs_decode_parser.h"

#define CUDA_DBG
#define CUDA_DBG1

#define FUNC_NAME_SIZE		200
#define MAX_LONG_LONG	200

static char *buffer ;
char pbuf[ FUNC_NAME_SIZE + 1 ] ;
static long long dbuf[ MAX_LONG_LONG + 2 ] ; // to handle double in coding, 2 per float

int gettype ( char **cp, int *cetype ) ;
void getint ( char **cp, int *cetype ) ;
void getsint ( char **cp, int *cetype ) ;
void getslonglong ( char **cp, long long *cetype ) ;
int getcode( const char *s, char **cp, char *code, int *size ) ;
int ck_bin ( int *bp, int size, int max ) ;

void p_QuantMeasurementsBasic( const char *, void *vcsp ) ;
void p_UniformQuantizer( const char *, void *vcsp ) ;
void p_VidRegion( const char *, void *vcsp ) ;
void p_RawVidInfo( const char *, void *vcsp ) ;
void p_SensingMatrixWH( const char *, void *vcsp ) ;
void p_CS_EncParams( const char *, void *vcsp ) ;

int get_more_data ( int fid, int size, char *buf, char **currp, int min_size, int reset ) ;
static int input_fid ; 

struct parser_func {
const char *name ;
int (*foo)( void *) ;
void (*pfoo)( const char *, void *) ;
int min_size ;
} ;

struct parser_func avail_parser_funcs[] = { 
	{ "CS_EncParams", get_CS_EncParams, p_CS_EncParams },
	{ "RawVidInfo", get_RawVidInfo, p_RawVidInfo },
	{ "VidRegion", get_VidRegion, p_VidRegion },
	{ "SensingMatrixWH", get_SensingMatrixWH, p_SensingMatrixWH },
	{ "UniformQuantizer", get_UniformQuantizer, p_UniformQuantizer },
	{ "QuantMeasurementsBasic", get_QuantMeasurementsBasic, p_QuantMeasurementsBasic }
} ;

static const int NUM_OF_FUNCS = int(sizeof( avail_parser_funcs ) / sizeof ( struct parser_func ));

#define FUNCS_IN_INPUT		10
struct parser_func parser_funcs[ FUNCS_IN_INPUT ] ;

static char *input_lp ;
static int data_file_size = 0 ;
static int data_file_used_size = 0 ;
static int data_read_size = 0 ;

/*
cs_decode_parser_init:
	for open and read from the csvid file, this might change
	after we have the network interface.
*/
int
cs_decode_parser_init( char *fname, int read_size )
{
	char *olp ;
	int k, jj, found, t, i ;
	struct stat *sp ;

	if (( input_fid = open ( fname, O_RDONLY )) < 0 )
	{
		printf("%s: ERR open %s failed %d \n", __func__, fname, errno ) ;
		return ( 0 ) ;
	}	   

	data_read_size = read_size ;
	
	printf("fid %d read %d\n", input_fid, data_read_size ) ;

	sp = ( struct stat * )dbuf ;
	if ( fstat ( input_fid, sp ) < 0 )
	{
		printf("%s: ERR stat failed %d \n", __func__, errno ) ;
		return ( 0 ) ;
	}

	data_file_size = sp->st_size ;

	if (!( buffer = ( char * )malloc ( data_read_size + 1000 )))
	{
		printf("%s: buffer malloc failed (%d): %s\n", __func__, errno, strerror(errno) ) ;
		return ( 0 ) ;
	}

	printf("%s: file %s size %d buffer %p\n", __func__, fname, data_file_size, buffer ) ;

	input_lp = buffer ;
	get_more_data ( input_fid, data_read_size, buffer, &input_lp, data_read_size, 1 ) ;

#ifdef CUDA_OBS 
	if (( i = read ( input_fid, buffer, BUFFER_SIZE )) < 0 )
	{
		printf("%s : ERR read %d failed \n", __func__, i ) ;
		return ( 0 ) ;
	}	   

	printf("%s: read %d buffer %p \n", __func__, i, buffer ) ;
#endif 

	gettype ( &input_lp, &t ) ;

	if ( t != 0 )
	{
		printf("cs_decode_parser_init: wrong type %d \n", t ) ;
		return ( 0 ) ;
	}

	getint ( &input_lp, &t ) ;

	olp = input_lp + t ;

	k = 1 ;
	while ( input_lp < olp )
	{
		getint ( &input_lp, &jj ) ;

		if ( jj > FUNC_NAME_SIZE )
		{
			printf("func name too long exit ... jj %d \n", jj ) ;
			return ( 0 ) ;
		}
		 
		strncpy ( pbuf, input_lp, (size_t)jj ) ;
		pbuf[jj] = 0 ;

		found = 0 ;
		for ( i = 0 ; i < NUM_OF_FUNCS ; i++ )
		{
			if ( !strcmp ( pbuf, avail_parser_funcs[ i ].name ))
			{
				parser_funcs[k].foo = avail_parser_funcs[ i ].foo ;
				parser_funcs[k++].pfoo = avail_parser_funcs[ i ].pfoo ;
				found++ ;
				break ;
			}
		}

		if ( k > FUNCS_IN_INPUT )
		{
			printf("too many funcs, exit ... k %d \n", k ) ;
			return ( 0 ) ;
		}

		if ( found )
			printf("%s : parser %s %d\n", __func__, pbuf, k-1 ) ;
		else
		{
			printf("%s : parser %s not found \n", __func__, pbuf) ;
			return ( 0 ) ;
		}

		input_lp += jj ;
	}

	printf("%s : found %d funcs \n", __func__, k - 1 ) ;

	return ( 1 ) ;
}

void
cs_decode_parser_reinit( int read_size )
{
#ifdef CUDA_DBG 
	printf("%s : new size %d \n", __func__, read_size ) ;
#endif 

	data_read_size = read_size ;

	free( buffer ) ;

	buffer = ( char * )malloc ( data_read_size + 2500 ) ;

	// input_lp = buffer ;

	get_more_data( input_fid, data_read_size, input_lp, &input_lp, 2500, 1 ) ;
}	

int
get_more_data ( int fid, int size, char *buf, char **currp, int min_size, int reset )
{
	int i ;
	static off_t last_file_pos = 0 ;
	static char *last_from = NULL ;
	static int last_size = 0 ;
	static int read_cnt = 0 ;

#ifdef CUDA_OBS 
	printf("%s : fid %d size %d buf %p cur %p %p lastsize %d last_from %p pos %d reset %d\n",
		__func__, fid, size, buf, currp, *currp, last_size, last_from, last_file_pos,
		reset ) ;
#endif 

	if ( last_from )
		i = *currp - last_from ; 	// used
	else
		i = 0 ;

#ifdef CUDA_OBS 
	printf("	used %d \n", i ) ;
#endif 

	if ( reset )
	{
	   	// last_file_pos = 0 ;
		last_from = buf ;
		last_size = 0 ;

		// lseek ( fid, last_file_pos, SEEK_SET ) ;
	}

#ifdef CUDA_OBS 
	printf("%s : fid %d size %d buf %p cur %p %p lastsize %d last_from %p pos %d reset %d\n",
		__func__, fid, size, buf, currp, *currp, last_size, last_from, last_file_pos,
		reset ) ;
#endif 

	data_file_used_size += i ;

	if ( data_file_used_size == data_file_size )
		return ( 0 ) ;

	if ( !reset )
	{
		if (( last_size - i ) > min_size )
			return ( last_size - i ) ;
	}

	last_file_pos += i ; 

#ifdef CUDA_OBS 
	printf("	last_file_pos %d \n", last_file_pos ) ;
#endif 

	
	if (i = lseek ( fid, last_file_pos, SEEK_SET ) < 0 )
	{
		printf("%s : ERR read %d failed \n", __func__, i ) ;
		return ( -1 ) ;
	}

#ifdef CUDA_OBS 
	printf("	last_file_pos seek %d \n", i ) ;
#endif 

	if (( last_size = read ( fid, buf, size )) < 0 )
	{
		printf("%s : ERR read %d failed \n", __func__, last_size ) ;
		return ( -1 ) ;
	}	   

	read_cnt++ ;

	last_from = buf ;
	*currp = buf ;

#ifdef CUDA_OBS 
	printf("%s: %d == read %d buffer %p pos %d cur %p\n", __func__, read_cnt,
		last_size, buf, last_file_pos, *currp ) ;
#endif 

	return ( last_size ) ;
}

int
get_next_type ( int *type )
{
	if ( !gettype ( &input_lp, type ))
		return ( -1 ) ;	// end of file

	if (( *type > 0 ) && ( *type <= NUM_OF_FUNCS ))
		return ( 1 ) ;
	else
	{
		printf("get_next_type: err type %d \n", *type ) ; 
		return ( 0 ) ;
	}
}

int
get_next_element ( int type, void *d )
{
	int i = 0 ;

#ifdef CUDA_OBS 
	printf("%s : type %d mem %p \n", __func__, type, d ) ;
#endif 

	if (( type > 0 ) && ( type <= NUM_OF_FUNCS ))
		i = parser_funcs[ type ].foo ( d ) ;
	else
		printf("%s : err type %d \n", __func__, type ) ;

	if ( i == 0 )
		printf("%s : type %d failed \n", __func__, type ) ;

	return ( i ) ;
}

int
p_element ( int type, char *s, void *d ) 
{
	int i = 0 ;

	if (( type > 0 ) && ( type <= NUM_OF_FUNCS ))
	{
		i++ ;
		parser_funcs[ type ].pfoo ( s, d ) ;
	}
	else
		printf("%s : err type %d \n", __func__, type ) ;

	return ( i ) ;
}


void
p_QuantMeasurementsBasic( const char *s, void *vcsp )
{
	struct QuantMeasurementsBasic *csp = ( struct QuantMeasurementsBasic * )vcsp ;

	printf("%s : %s :nbin %d noclip %d lenb %d lens %d mean %f stdv %f \n",
		__func__,
		s,
		csp->nbin,
		csp->noclip,
		csp->lenb,
		csp->lens,
		csp->mean_msr,
		csp->stdv_msr ) ;
}

int
get_QuantMeasurementsBasic( void *vcsp )
{
	struct QuantMeasurementsBasic *csp = ( struct QuantMeasurementsBasic * )vcsp ;
	int i, leng ;
	char *olp ;

	get_more_data ( input_fid, data_read_size, buffer, &input_lp, data_read_size, 0 ) ;

	getint( &input_lp, &leng ) ;

	olp = input_lp + leng ; 

	getint ( &input_lp, &csp->nbin ) ;
	getint ( &input_lp, &csp->noclip ) ;
	getint ( &input_lp, &csp->lenb ) ;
	getint ( &input_lp, &csp->lens ) ;

	for ( i = 0 ; i < NUM_QuantMeasurementsBasic_FLOAT * 2 ; i++ )
		getslonglong ( &input_lp, &dbuf[i] ) ;

	csp->mean_msr = (float) ( dbuf[0] * pow (( double )2, dbuf[NUM_QuantMeasurementsBasic_FLOAT] )) ;
	csp->stdv_msr = (float) ( dbuf[1] * pow (( double )2, dbuf[1+NUM_QuantMeasurementsBasic_FLOAT] )) ;

	for ( i = 0 ; i < csp->lenb ; i++ )
		getsint( &input_lp, &csp->h_msr_idxp[i] ) ;

	// do some checking ... LDL ck parser.c

	if ( olp != input_lp )
	{
		printf("%s: olp %p lp %p \n", __func__, olp, input_lp ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 
	p_QuantMeasurementsBasic( "get_QuantMeasurementsBasic", vcsp ) ;

#endif 
	return ( 1 ) ;
}

void
p_UniformQuantizer( const char *s, void *vcsp )
{
	struct UniformQuantizer *csp = ( struct UniformQuantizer * )vcsp ;

	printf("%s : %s : save %d wdth_mlt %f wdth_unit %f ampl %f\n",
		__func__,
		s,
		csp->save_clipped,
		csp->q_wdth_mltplr,
		csp->q_wdth_unit,
		csp->q_ampl_mltplr ) ;
}

int
get_UniformQuantizer( void *vcsp )
{
	struct UniformQuantizer *csp = ( struct UniformQuantizer * )vcsp ;
	int i, leng ;
	char *olp ;

	get_more_data ( input_fid, data_read_size, buffer, &input_lp, sizeof ( *csp ), 0 ) ;

	getint( &input_lp, &leng ) ;

	olp = input_lp + leng ; 

	getint ( &input_lp, &csp->save_clipped ) ;

	for ( i = 0 ; i < NUM_UniformQuantizer_FLOAT * 2 ; i++ )
		getslonglong ( &input_lp, &dbuf[i] ) ;

	csp->q_wdth_mltplr = (float) ( dbuf[0] * pow (( double )2, dbuf[NUM_UniformQuantizer_FLOAT] )) ;
	csp->q_wdth_unit = (float) ( dbuf[1] * pow (( double )2, dbuf[1+NUM_UniformQuantizer_FLOAT] )) ;
	csp->q_ampl_mltplr = (float) ( dbuf[2] * pow (( double )2, dbuf[2+NUM_UniformQuantizer_FLOAT] )) ;

	if ( olp != input_lp )
	{
		printf("%s: olp %p lp %p \n", __func__, olp, input_lp ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 
	p_UniformQuantizer( "get_UniformQuantizer", vcsp ) ;
#endif 
	return ( 1 ) ;
}

void
p_VidRegion( const char *s, void *vcsp )
{
	struct VidRegion *csp = ( struct VidRegion * )vcsp ;

	printf("%s : %s : n_blk %d v %d h %d t %d \n",
		__func__,
		s,
		csp->n_blk,
		csp->blk_v,
		csp->blk_h,
		csp->blk_t ) ; 
}

int
get_VidRegion( void *vcsp )
{
	struct VidRegion *csp = ( struct VidRegion * )vcsp ;
	int leng ;
	char *olp ;

	get_more_data ( input_fid, data_read_size, buffer, &input_lp, sizeof ( *csp ), 0 ) ;

	getint( &input_lp, &leng ) ;

	olp = input_lp + leng ; 

	getint ( &input_lp, &csp->n_blk ) ;
	getint ( &input_lp, &csp->blk_v ) ;
	getint ( &input_lp, &csp->blk_h ) ;
	getint ( &input_lp, &csp->blk_t ) ;

	if ( olp != input_lp )
	{
		printf("%s: olp %p lp %p \n", __func__, olp, input_lp ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 
	p_VidRegion( "get_VidRegion", vcsp ) ;
#endif 
	return ( 1 ) ;
}

void
p_RawVidInfo( const char *s, void *vcsp )
{
	struct RawVidInfo *csp = ( struct RawVidInfo * ) vcsp ;

	printf("%s : %s : uv_pre %d prec %d n_frame %d width %d height %d seg %d segfr %d fps %d uv %d %d %d\n",
		__func__,
		s,
		csp->UV_present,
		csp->precision,
		csp->n_frames,
		csp->width,
		csp->height,
		csp->seg_start_frame,
		csp->seg_n_frames,
		csp->fps,
		csp->uv_ratio[0],
		csp->uv_ratio[1],
		csp->uv_ratio[2] ) ;
}

int
get_RawVidInfo( void *vcsp )
{
	struct RawVidInfo *csp = ( struct RawVidInfo * ) vcsp ;
	int leng ;
	char *olp ;

	get_more_data ( input_fid, data_read_size, buffer, &input_lp, sizeof ( *csp ), 0 ) ;

	getint( &input_lp, &leng ) ;

	olp = input_lp + leng ; 

	getint ( &input_lp, &csp->UV_present ) ;
	getint ( &input_lp, &csp->precision ) ;
	getint ( &input_lp, &csp->n_frames ) ;
	getint ( &input_lp, &csp->width ) ;
	getint ( &input_lp, &csp->height ) ;
	getint ( &input_lp, &csp->seg_start_frame ) ;
	getint ( &input_lp, &csp->seg_n_frames ) ;
	getint ( &input_lp, &csp->fps ) ;

	if ( csp->UV_present )
	{
		getint ( &input_lp, &csp->uv_ratio[0] ) ;
		getint ( &input_lp, &csp->uv_ratio[1] ) ;
		getint ( &input_lp, &csp->uv_ratio[2] ) ;
	}

	if ( olp != input_lp )
	{
		printf("%s: olp %p lp %p \n", __func__, olp, input_lp ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 
	p_RawVidInfo( "get_RawVidInfo",  vcsp ) ;
#endif 

	return ( 1 ) ;
}

void
p_SensingMatrixWH( const char *s, void *vcsp )
{
	struct SensingMatrixWH *csp = ( struct SensingMatrixWH * )vcsp ;

	printf("%s : %s : row %d col %d trans %d seed %d sqr %d\n",
		__func__,
		s,
		csp->n_rows,
		csp->n_cols,
		csp->is_transposed,
		csp->seed,
		csp->sqr_order ) ;
}

int
get_SensingMatrixWH( void *vcsp )
{
	struct SensingMatrixWH *csp = ( struct SensingMatrixWH * )vcsp ;
	char *olp ;
	int j, leng ;

	get_more_data ( input_fid, data_read_size, buffer, &input_lp, sizeof ( *csp ), 0 ) ;

	getint( &input_lp, &leng ) ;

	olp = input_lp + leng ; 

	getint ( &input_lp, &csp->n_rows ) ;
	getint ( &input_lp, &csp->n_cols ) ;
	getint ( &input_lp, &csp->is_transposed ) ;
	getint ( &input_lp, &csp->seed ) ;

	j = SensingMatrixWH_code_len ;
	if ( !getcode( "code", &input_lp, csp->code, &j ))
		return ( 0 ) ;

	getint ( &input_lp, &csp->sqr_order ) ;

	if ( olp != input_lp )
	{
		printf("%s: olp %p lp %p \n", __func__, olp, input_lp ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 

	p_SensingMatrixWH( "get_SensingMatrixWH", vcsp ) ;

#endif 

	return ( 1 ) ;
}

void 
p_CS_EncParams( const char *s, void *vcsp )
{
	struct CS_EncParams *csp = ( struct CS_EncParams *)vcsp ;

	printf("%s : %s : n_fr %d start %d seed %d color %d wnd %d action %d los_coder %d \n",
		__func__,
		s, 
		csp->n_frames,
		csp->start_frame,
		csp->random_seed,
		csp->process_color,
		csp->wnd_type,
		csp->qntzr_outrange_action,
		csp->lossless_coder ) ;

	printf(" random spa %d temp %d blk %d %d %d ratio %f width %f ampl %f\n",
		csp->random_rpt_spatial,
		csp->random_rpt_temporal,
		csp->blk_size.v,
		csp->blk_size.h,
		csp->blk_size.t,
		csp->msrmnt_input_ratio,
		csp->qntzr_wdth_mltplr,
		csp->qntzr_ampl_stddev ) ;
}

int 
get_CS_EncParams( void *vcsp )
{
	struct CS_EncParams *csp = ( struct CS_EncParams *)vcsp ;
	char *olp ;
	int k, leng ;

	get_more_data ( input_fid, data_read_size, buffer, &input_lp, sizeof ( *csp ), 0 ) ;

	getint( &input_lp, &leng ) ;

	olp = input_lp + leng ; 

	getsint( &input_lp, &csp->n_frames ) ;

	getint( &input_lp, &csp->start_frame ) ;

	getint( &input_lp, &csp->random_seed ) ;

	getint( &input_lp, &csp->process_color ) ;

	getint( &input_lp, &csp->wnd_type ) ;

	getint( &input_lp, &csp->qntzr_outrange_action ) ;

	getint( &input_lp, &csp->lossless_coder ) ;

	getint( &input_lp, &csp->conv_mode ) ;

	getint( &input_lp, &csp->leng_conv_rng ) ;

	getint( &input_lp, &csp->random_rpt_spatial ) ;

	getint( &input_lp, &csp->random_rpt_temporal ) ;

	getint( &input_lp, &csp->conv_rng.v ) ;
	getint( &input_lp, &csp->conv_rng.h ) ;
	getint( &input_lp, &csp->conv_rng.t ) ;

	getint( &input_lp, &csp->blk_size.v ) ;
	getint( &input_lp, &csp->blk_ovrlp.v ) ;
	getint( &input_lp, &csp->zero_ext_b.v ) ;
	getint( &input_lp, &csp->zero_ext_f.v ) ;
	getint( &input_lp, &csp->wrap_ext.v ) ;
	getint( &input_lp, &csp->blk_pre_diff.v ) ;

	getint( &input_lp, &csp->blk_size.h ) ;
	getint( &input_lp, &csp->blk_ovrlp.h ) ;
	getint( &input_lp, &csp->zero_ext_b.h ) ;
	getint( &input_lp, &csp->zero_ext_f.h ) ;
	getint( &input_lp, &csp->wrap_ext.h ) ;
	getint( &input_lp, &csp->blk_pre_diff.h ) ;

	getint( &input_lp, &csp->blk_size.t ) ;
	getint( &input_lp, &csp->blk_ovrlp.t ) ;
	getint( &input_lp, &csp->zero_ext_b.t ) ;
	getint( &input_lp, &csp->zero_ext_f.t ) ;
	getint( &input_lp, &csp->wrap_ext.t ) ;
	getint( &input_lp, &csp->blk_pre_diff.t ) ;

	getint( &input_lp, &csp->case_no ) ;
	getint( &input_lp, &csp->n_cases ) ;

	for ( k = 0 ; k < NUM_CS_EncParams_FLOAT * 2 ; k++ )
		getslonglong ( &input_lp, &dbuf[k] ) ;

	csp->msrmnt_input_ratio = (float) ( dbuf[0] * pow (( double )2, dbuf[4] )) ;
	csp->qntzr_wdth_mltplr = (float) ( dbuf[1] * pow (( double )2,
		dbuf[1+NUM_CS_EncParams_FLOAT] )) ;

	csp->qntzr_ampl_stddev = (float) ( dbuf[2] * pow (( double )2,
		dbuf[2+NUM_CS_EncParams_FLOAT] )) ;

	csp->lossless_code_AC_gaus_thrsh = (float) ( dbuf[3] * pow (( double )2,
		dbuf[3+NUM_CS_EncParams_FLOAT] )) ;

	k = CS_EncParams_code_len ;
	if ( !getcode( "msrmnt_mtrx", &input_lp, csp->msrmnt_mtrx_code, &k ))
		return ( 0 ) ;

	printf("code length %d \n", k ) ;

	if ( olp != input_lp )
	{
		printf("%s: olp %p lp %p \n", __func__, olp, input_lp ) ;
		return ( 0 ) ;
	}
#ifdef CUDA_DBG 

	p_CS_EncParams( "get_CS_EncParams", vcsp ) ;

#endif 

	return ( 1 ) ;
}


// misc supporting routines ...

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

int
gettype ( char **cp, int *cetype )
{
	char *lcp ;
	int i ;

	i = get_more_data ( input_fid, data_read_size, buffer, cp, 1000, 0 ) ;

	if ( !i )
	{
		printf("%s : END OF FILE \n", __func__ ) ;
		return ( 0 ) ;
	}

	lcp = *cp ;
	i = *lcp++ ;
	*cetype = i & 0xff ;

#ifdef CUDA_OBS 
	printf("%s: lcp %p *cp %p *cetype %d \n", __func__, lcp, *cp, *cetype ) ;
#endif 

	*cp = lcp ;

	return ( 1 ) ;
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

int
getcode( const char *s, char **cp, char *code, int *size )
{
	char *lcp ;
	int i ;
	int first = 1 ;
	int max ;

	max = *size ;

	getint( cp, &i ) ;


	if ( i > max )
	{
		printf("%s: ERR cp %p len %d max %d\n", __func__, *cp, i, max ) ;
		return ( 0 ) ;
	}

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

	return ( 1 ) ;
}
