#include <iostream>
using namespace std;

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// rand() from matlab

#include "RndC_ifc.h"
#include "RndCState.h"

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_header.h"
#include "cs_block.h"
#include "cs_perm_mlseq.h"
#include "cs_expand.h"
#include "cs_interpolate.h"
#include "cs_perm_selection.h"
#include "cs_copy_box.h"
#include "cs_edge_detect_v2.h"
#include "cs_motion_detect_v2.h"
#include "cs_motion_report.h"
#include "cs_ipcam.h"
#include "cs_config.h"
#include "cs_buffer.h"
#include "cs_dct.h"
#include "cs_random.h"
#include "cs_quantize.h"
#include "cs_perm_generic.h"
#include "cs_decode_parser.h"
#include "cs_domultivec.h"
#include "cs_matrix.h"
#include "cs_decode_misc.h"
#include "cs_vector.h"
#include "cs_sparser.h"
#include "cs_complgrngn.h"

#include "recon.h"

#define CUDA_DBG 

#define DBG_CP_DOWN		0x1
#define DBG_BLKING		0x2
#define DBG_WHM			0x4
#define DBG_PERM_R		0x8
#define DBG_PERM_L		0x10
#define DBG_INTER		0x20
#define DBG_SWAP		0x40
#define DBG_EXPAND		0x80
#define DBG_ANALYSIS		0x100
#define DBG_ED		0x200
#define DBG_MOTION		0x400
#define DBG_L1_NORM		0x800
#define DBG_COPY_DONE		0x1000
#define DBG_MT_IDX		0x2000
#define DBG_MT_STEP0		0x4000
#define DBG_MT_STEP1		0x8000
#define DBG_MT_STEP2		0x10000
#define DBG_MT_STEP3		0x20000
#define DBG_MT_STEP4		0x40000
#define DBG_C_2_I			0x80000


enum {
	CS_TIMER_TOTAL,
	CS_TIMER_MEMCPY_DOWN,
	CS_TIMER_C_TO_I,
	CS_TIMER_EXPANSION,
	CS_TIMER_INTER,
	CS_TIMER_BLOCKING,
	CS_TIMER_PERMR,
	CS_TIMER_MEA,
	CS_TIMER_PERML,
	CS_TIMER_SWAP,
	CS_TIMER_MEMCPY_UP,
	CS_TIMER_ANALYSIS,
	CS_TIMER_ANALYSIS_EDGE,
	CS_TIMER_ANALYSIS_MD0,
	CS_TIMER_ANALYSIS_MD1,
	CS_TIMER_ANALYSIS_MD2,
	CS_TIMER_ANALYSIS_MD3,
	CS_TIMER_ANALYSIS_MD4,
	CS_TIMER_COUNT
} ;

static char *timer_name[] = {
	"timer total",
	"memcpy to device",
	"expand c to i",
	"expansion",
	"interpolation",
	"blocking",
	"perm R",
	"measurement",
	"perm L",
	"swap",
	"memcpy to host",
	"analysis",
	"analysis edge",
	"analysis md0",
	"analysis md1",
	"analysis md2",
	"analysis md3",
	"analysis md4",
	"the end"
} ;

static struct CS_EncParams _rc_enc, *rc_enc_p = &_rc_enc ;
static struct RawVidInfo _rc_raw, *rc_raw_p = &_rc_raw ;
static struct VidRegion _rc_vid, *rc_vid_p = &_rc_vid ;
static struct SensingMatrixWH _rc_sens, *rc_sens_p = &_rc_sens ;
static struct UniformQuantizer _rc_quan, *rc_quan_p = &_rc_quan ;
static struct QuantMeasurementsBasic _rc_mea, *rc_mea_p = &_rc_mea ;

void *rc_cep[] = {
	(void *)NULL,
	(void *)rc_enc_p,	// 1
	(void *)rc_raw_p,
	(void *)rc_vid_p,
	(void *)rc_sens_p,
	(void *)rc_quan_p,
	(void *)rc_mea_p
} ;

#define MAX_ELEMENT_SIZE		10000	// i.e. QuantMeasurementsBasic 

// proto

int get_some_code_elements( int from, int to ) ;

// buf in device

#define BUF_DESC_CNT	3
#define SIZE_VHTC		0
#define SIZE_MOD2_VHTC	1
#define SIZE_VHTC_X2	2

#define NUM_SIZE_VHTC		10
#define NUM_SIZE_MOD2_VHTC	10
#define NUM_SIZE_VHTC_X2	10

struct cs_buf_desc cs_buf_desc[BUF_DESC_CNT];

static int mod2_size, vhtc_size, vhtc_x2_size, lenb_size ;

// random number 

// eps 

struct CS_DecParams CS_DecParams = {
	0.1,
	0.5,
	0.05,
	3,
	2,
	0.2,
	0.01,

	// beta
	2, 
	0.2,
	0.1,
	0.1,
	0,	// Inf
	0	// Inf
	
} ;

struct eps eps ;
struct beta beta ;
struct lambda d_lambda ;
float *d_xvec ;
struct grad_x_const d_grad_x_const ;
struct xerr d_xerr ;

struct rc_SensingMatrixWH rc_SensingMatrixWH ;

void
pusage( char *s )
{
	printf("Usage: %s -f csvid\n", s ) ;
}

static int *d_l_permp = NULL ;	// left permutation table on device
static int *d_il_permp = NULL ;	// inverse left permutation table on device
static int *d_r_permp = NULL ; // right permutation table on device
static int *h_permp = NULL ; // permutaion table on host
static float *d_multivec = NULL ; // output of h_do_multi_vec

static float *d_wvec_ref ;
static float *d_wvec ;

// zeroed rows

static int *h_zeroed_rows ; 
static int *d_zeroed_rows ;
static int num_of_zeroed_rows ;

// tmp buffer on device

static float *d_f_tbuf_1, *d_f_tbuf_2 ;

// misc

static int blk_in_x, blk_in_y ; 

// proto

void init_solver_eps ( struct eps *eps_p, int num_of_meas, struct CS_DecParams *opt_p,
	float q_maxerr, float q_stdv_err, int n_sprsvec  ) ;
void init_solver_beta( struct beta *betap, struct CS_DecParams *opt_p,
	struct rc_SensingMatrixWH *smhp ) ;

main( int ac, char *av[] )
{
	char *cp, opt ;
	char *csvid = NULL ;
	int i, *ip ;
	int blk_cnt, dbg_cnt ;
	static RndC_uint32 h_seed ;

	float ampl, intvl ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	while ((opt = getopt(ac, av, "f:")) != -1) 
	{
		printf(" opt %c \n", opt ) ;

		switch (opt) {
		case 'f' :
			csvid = optarg ;
			break ;
		}
	}

	if ( csvid == NULL )
	{
		pusage( av[0] ) ;
		exit ( 1 ) ;
	}

	if ( !dbg_init ( 1000000 ))
	{
		fprintf( stderr, "dbg_init : failed \n" ) ;
		exit( 2 ) ;
	}

	if ( !cs_decode_parser_init ( csvid, MAX_ELEMENT_SIZE ))
	{
		fprintf( stderr, "!cs_decode_parser_init: failed \n" ) ;
		exit( 2 ) ;
	}

	if ( !get_some_code_elements( 1, 5 ))
	{
		fprintf( stderr, "!get_some_code_elements: failed \n" ) ;
		exit( 2 ) ;
	}

	// begin : init per block 

	// init 5/UniformQuantizer

	rc_quan_p->q_wdth = rc_quan_p->q_wdth_mltplr * rc_quan_p->q_wdth_unit ;

	// init 4/SensingMatrixWH

	h_seed = rc_sens_p->seed ;

	// end : init per block 

	i = rc_sens_p->n_rows * 2 ; 

	cs_decode_parser_reinit ( i ) ;

	if (!( ip = ( int * )malloc ( i * sizeof ( int ))))
	{
		printf("malloc ip failed \n") ;
		exit( 4 ) ;
	}

	rc_mea_p->h_msr_idxp = ip ;

	if ( !get_some_code_elements( 6, 6 ))
	{
		fprintf( stderr, "get_some_code_elements: 6 failed \n" ) ;
		exit( 2 ) ;
	}

	lenb_size = rc_mea_p->lenb ;

	blk_in_x = rc_raw_p->height / rc_enc_p->blk_size.v ;
	blk_in_y = rc_raw_p->width / rc_enc_p->blk_size.h ;

	printf("blk_in_x %d y %d h %d w %d v %d h %d \n",
		blk_in_x,
		blk_in_y,
		rc_raw_p->height,
	   	rc_raw_p->width,
		rc_enc_p->blk_size.v,
		rc_enc_p->blk_size.h ) ;

	i = rc_enc_p->blk_size.v * rc_enc_p->blk_size.h * rc_enc_p->blk_size.t *
		(( rc_raw_p->UV_present )? 3 : 1 ) ;  

	printf("blk %d %d %d BLK size %d \n", 
		rc_enc_p->blk_size.v, rc_enc_p->blk_size.h, rc_enc_p->blk_size.t, i ) ;

	mod2_size = max_log2( i ) ;
	vhtc_size = i ;
	vhtc_x2_size = i << 1 ;

	cs_buf_desc[ SIZE_VHTC ].size = i * sizeof ( int ) ;
	cs_buf_desc[ SIZE_VHTC ].unit_size = i ;
	cs_buf_desc[ SIZE_MOD2_VHTC ].size = mod2_size * sizeof ( int ) ;
	cs_buf_desc[ SIZE_MOD2_VHTC ].unit_size = mod2_size ;
	cs_buf_desc[ SIZE_VHTC_X2 ].size = ( i << 1 ) * sizeof ( int ) ;
	cs_buf_desc[ SIZE_VHTC_X2 ].unit_size = ( i << 1 ) ;

	rc_SensingMatrixWH.sqr_order = mod2_size ;

	cs_buf_desc[ SIZE_VHTC ].cnt = NUM_SIZE_VHTC ;
	cs_buf_desc[ SIZE_MOD2_VHTC ].cnt = NUM_SIZE_MOD2_VHTC ;
	cs_buf_desc[ SIZE_VHTC_X2 ].cnt = NUM_SIZE_MOD2_VHTC ;

	if ( !cs_buffer_init ( cs_buf_desc, BUF_DESC_CNT, 0 ))
	{
		fprintf( stderr, "!cs_buffer_init: failed \n" ) ;
		exit( 4 ) ;
	}

	h_zeroed_rows = ( int * ) malloc ( i * sizeof ( int )) ;
	d_zeroed_rows = ( int * ) cs_get_free_list ( SIZE_VHTC ) ;

	d_l_permp = ( int * )cs_get_free_list( SIZE_MOD2_VHTC ) ;
	d_il_permp = ( int * )cs_get_free_list( SIZE_MOD2_VHTC ) ;
	d_r_permp = ( int * )cs_get_free_list( SIZE_MOD2_VHTC ) ;
	h_permp = ( int * )malloc ( mod2_size * sizeof ( int )) ;

	d_wvec_ref = ( float * )cs_get_free_list( SIZE_VHTC_X2 ) ;
	d_wvec = ( float * )cs_get_free_list( SIZE_VHTC_X2 ) ;

	rc_mea_p->d_msr_p = ( float * )cs_get_free_list ( SIZE_MOD2_VHTC ) ;
	d_multivec = ( float * )cs_get_free_list ( SIZE_MOD2_VHTC ) ;

	d_xerr.d_A = ( float * )cs_get_free_list ( SIZE_VHTC ) ;
	d_xerr.d_D = ( float * )cs_get_free_list ( SIZE_VHTC_X2 ) ;

	if (( d_l_permp == NULL ) || ( d_r_permp == NULL ) || ( h_permp == NULL ) ||
		( d_il_permp == NULL ) || ( d_wvec_ref == NULL )|| ( rc_mea_p->d_msr_p == NULL ) 
		|| ( h_zeroed_rows == NULL ) || ( d_zeroed_rows == NULL ) ||
		( d_wvec == NULL ) ||
		( d_xerr.d_A == NULL ) || ( d_xerr.d_D == NULL))
	{
		fprintf( stderr, "perm buffer failed l %p r %p h %p \n", d_l_permp, d_r_permp, h_permp ) ;
		fprintf( stderr, "perm buffer failed d_msr %p zero h %p d %p\n", rc_mea_p->d_msr_p,
			h_zeroed_rows, d_zeroed_rows ) ;
		fprintf( stderr, "d_wvec_ref %p d_wvec %p d_xerr.d_A %p d_xerr.d_D %p\n",
			d_wvec_ref, d_wvec, d_xerr.d_A, d_xerr.d_D ) ;
		exit( 4 ) ;
	}
	fprintf( stderr, "perm buffer %p r %p il %p \n", d_l_permp, d_r_permp, d_il_permp ) ;

	blk_cnt = 0 ;
	dbg_cnt = 1 ;
	while ( dbg_cnt-- )
	{
		blk_cnt++ ;

		if ( lenb_size != rc_mea_p->lenb )
		{
			printf("%s : err blk_cnt %d lenb %d rc_lenb %d mismatch\n", __func__, blk_cnt,
				lenb_size, rc_mea_p->lenb ) ;
			exit ( 4 ) ;
		}

		// update the L/R permutation table

		if ( h_seed != rc_sens_p->seed )
		{
			fprintf( stderr, "seed : err blk_cnt %d want %d got %d \n",
				blk_cnt, h_seed, rc_sens_p->seed ) ;
			exit( 4 ) ;
		}

		// has to be L first ...

		if ( !h_set_random_table(( RndC_uint32 )h_seed, ( RndC_uint32 * )h_permp,
			( RndC_uint32 * )d_l_permp, mod2_size, 1, 1 ))
		{
			fprintf( stderr, "h_set_random_table: l blk_cnt %d \n", blk_cnt ) ;
			exit( 4 ) ;
		}

		dbg_p_d_data_i("l perm first 10", ( int * )d_l_permp, 10 ) ;

		h_do_permutation_generic_inverse( d_il_permp, d_l_permp, mod2_size ) ;

		if ( !h_set_random_table(( RndC_uint32 )h_seed, ( RndC_uint32 *)h_permp,
			( RndC_uint32 *)d_r_permp, mod2_size, 0, 0 ))
		{
			fprintf( stderr, "h_set_random_table: r blk_cnt %d \n", blk_cnt ) ;
			exit( 4 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_i("l perm :::", ( int * )d_l_permp, mod2_size ) ;
		dbg_p_d_data_i("il perm :::", ( int * )d_il_permp, mod2_size ) ;
		dbg_p_d_data_i("r perm :::", ( int * )d_r_permp, mod2_size ) ;
#endif 

#ifdef CUDA_DBG 
		dbg_p_d_data_i("l perm first 10", ( int * )d_l_permp, 10 ) ;
		dbg_p_d_data_i("il perm first 10", ( int * )d_il_permp, 10 ) ;
		dbg_p_d_data_i("r perm first 10", ( int * )d_r_permp, 10 ) ;

		dbg_p_d_data_i("l perm last 10", ( int * )d_l_permp + mod2_size - 10, 10 ) ;
		dbg_p_d_data_i("il perm last 10", ( int * )d_il_permp + mod2_size - 10, 10 ) ;
		dbg_p_d_data_i("r perm last 10", ( int * )d_r_permp + mod2_size - 10, 10 ) ;
#endif 

		// interpolate of u/v ... LDL do anything for decode?  maybe not ...

		// unquantize ...

		printf("UNQUANTIZE -- %d ---------------------------------------------------------\n", blk_cnt ) ;

		// copy the measurements to gpu

		put_d_data_i(( int * )rc_mea_p->d_msr_p, rc_mea_p->h_msr_idxp, rc_mea_p->lenb * sizeof ( int )) ;

		// -- first adjust the index of measurements

		intvl = rc_quan_p->q_wdth ; 
		i = rc_mea_p->nbin / 2 ;
		ampl = (( float(i) + 0.5 )) * intvl ;

#ifdef CUDA_DBG
		printf("unquantize : ampl %f intv %f \n", ampl, intvl ) ;
#endif 

		// quantize bin adjustment ---

		h_do_unquan_adj_index ( ( int * )rc_mea_p->d_msr_p, rc_mea_p->lenb, rc_mea_p->noclip,
			i - 1, rc_mea_p->nbin ) ; 

		/* DO FOR NO CLIP ... RAZI ???
		h_do_unquan_adj_index ( ( int * )rc_mea_p->d_msr_p + rc_mea_p->noclip, rc_mea_p->lenb -
			rc_mea_p->noclip, i - 1, rc_mea_p->nbin ) ; 
		*/

		// set up the zeroed_rows ---

		num_of_zeroed_rows = h_ck_bin ( ( int * )rc_mea_p->d_msr_p, rc_mea_p->lenb,
			rc_mea_p->nbin, h_zeroed_rows, rc_mea_p->noclip ) ;

		put_d_data_i ( d_zeroed_rows, h_zeroed_rows, num_of_zeroed_rows * sizeof ( int )) ;

#ifdef CUDA_DBG 
		dbg_p_d_data_i("zeroed rows", d_zeroed_rows, num_of_zeroed_rows ) ;
#endif 

		// unquantize ---

		h_do_int_to_float ( ( int * )rc_mea_p->d_msr_p, rc_mea_p->d_msr_p, rc_mea_p->lenb ) ;

		/* DO FOR NO CLIP ... RAZI ???
		h_do_unquan_msrmnts( rc_mea_p->d_msr_p + rc_mea_p->noclip, rc_mea_p->lenb -
			rc_mea_p->noclip, ampl, intvl, rc_mea_p->mean_msr ) ;
		*/

		h_do_unquan_msrmnts( rc_mea_p->d_msr_p, rc_mea_p->lenb, ampl, intvl, rc_mea_p->mean_msr ) ;

		// init eps ---

		i = ( rc_enc_p->blk_size.v - 1 ) * rc_enc_p->blk_size.h * rc_enc_p->blk_size.t +
			rc_enc_p->blk_size.v * ( rc_enc_p->blk_size.h - 1 ) * rc_enc_p->blk_size.t ;

		if ( rc_enc_p->process_color )
			i *= 3 ;	// 3 colors 

		init_solver_eps ( &eps, rc_mea_p->lenb, &CS_DecParams,
			rc_quan_p->q_wdth / 2.0, rc_quan_p->q_wdth_unit / powf ( 12.0, 0.5 ), i ) ;

		// init beta ---

		init_solver_beta( &beta, &CS_DecParams, &rc_SensingMatrixWH ) ;

		// init_solver_xvec ---
		// init d_xvec :: has to increase the size to SIZE_MOD2_VHTC from SIZE_VHTC, since
		// we need that in permutation and walsh hadamand ... like in h_do_multi_vec() 

		d_xvec = ( float * )cs_get_free_list ( SIZE_MOD2_VHTC ) ;
		clear_device_mem ( d_xvec, cs_buf_desc[ SIZE_MOD2_VHTC ].unit_size ) ;

		// MATLAB d_xvec partially VERIFIED ...
		// MATLAB VERIFIED ...

		// init lambda ---

		d_lambda.d_A = ( float * )cs_get_free_list ( SIZE_VHTC ) ;
		clear_device_mem ( d_lambda.d_A, cs_buf_desc[ SIZE_VHTC ].unit_size ) ;
		d_lambda.A_size = rc_mea_p->lenb ;

		d_lambda.d_D = ( float * )cs_get_free_list ( SIZE_VHTC_X2 ) ;
		clear_device_mem ( d_lambda.d_D, cs_buf_desc[ SIZE_VHTC_X2 ].unit_size ) ;
		d_lambda.D_size = cs_buf_desc[ SIZE_VHTC_X2 ].unit_size ;	// 38016 x 2 ... 76032, not 75072

		// init grad_x_const ---

		d_grad_x_const.d_A = ( float * )cs_get_free_list ( SIZE_VHTC ) ;
		clear_device_mem ( d_grad_x_const.d_A, cs_buf_desc[ SIZE_VHTC ].unit_size ) ;

		d_grad_x_const.d_D = ( float * )cs_get_free_list ( SIZE_VHTC ) ;
		clear_device_mem ( d_grad_x_const.d_D, cs_buf_desc[ SIZE_VHTC ].unit_size ) ;

		d_grad_x_const.d_sum = ( float * )cs_get_free_list ( SIZE_VHTC ) ;
		clear_device_mem ( d_grad_x_const.d_sum, cs_buf_desc[ SIZE_VHTC ].unit_size ) ;

#ifdef CUDA_DBG 
		dbg_p_d_data_f("d_grad_x_const.d_sum first 10", d_grad_x_const.d_sum, 10 ) ;
		dbg_p_d_data_f("d_grad_x_const.d_sum last 10", d_grad_x_const.d_sum + 
			cs_buf_desc[ SIZE_VHTC ].unit_size - 10, 10 ) ;
#endif 

		// init xerr ---

		if (( d_f_tbuf_1 = ( float * ) cs_get_free_list ( SIZE_MOD2_VHTC )) == NULL )
		{
			printf("init xerr buf err \n") ;
			exit ( 3 ) ;
		}

		// do A ... no need to zero out d_f_tbuf_1 ...

		// moved up d_xerr.d_A = ( float * )cs_get_free_list ( SIZE_VHTC ) ;
		clear_device_mem ( d_xerr.d_A, cs_buf_desc[ SIZE_VHTC ].unit_size ) ;
		d_xerr.A_size = rc_mea_p->lenb ;

		h_do_multi_vec( d_xvec, d_multivec, d_f_tbuf_1, d_r_permp, d_il_permp, vhtc_size, lenb_size ) ;

		h_do_vector_zero_some( d_multivec, d_zeroed_rows, num_of_zeroed_rows ) ;

		h_do_vector_sub_vector( d_multivec, rc_mea_p->d_msr_p, d_xerr.d_A, lenb_size ) ;

		cs_put_free_list(( char *)d_f_tbuf_1, SIZE_MOD2_VHTC ) ;

		// do D ...

		// moved up d_xerr.d_D = ( float * )cs_get_free_list ( SIZE_VHTC_X2 ) ;
		clear_device_mem ( d_xerr.d_D, cs_buf_desc[ SIZE_VHTC_X2 ].unit_size ) ;
		d_xerr.D_size = cs_buf_desc[ SIZE_VHTC_X2 ].unit_size ;	// 38016 x 2 ... 76032, not 75072

		d_xerr.J = 0.0 ;

#ifdef CUDA_DBG 
		dbg_pr_first_last ("d_xerr.d_A", d_xerr.d_A, lenb_size, 10 ) ;
#endif 

		// init optimize_solver_w

#ifdef CUDA_OBS 
		// before switch to DEMO ...

		clear_device_mem ( d_wvec_ref, cs_buf_desc[ SIZE_VHTC_X2 ].unit_size ) ;

		h_optimize_solver_w ( d_xvec, beta.D, lambda.d_D, d_wvec_ref, &d_xerr.J, 
			d_wvec, d_xerr.d_D, 

#endif 

// QQQ

		/*
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		 ****
		*/




		// one DONE ...

		printf("blk %d DONE \n", blk_cnt ) ;

		// start the next loop ... LDL

		if ( !get_some_code_elements( 3, 6 ))
		{
			fprintf( stderr, "!get_some_code_elements: failed \n" ) ;
			exit( 2 ) ;
		}

		// begin : init per block 

		// init 4/SensingMatrixWH 

		h_seed = rc_sens_p->seed ;

		// init 5/UniformQuantizer

		rc_quan_p->q_wdth = rc_quan_p->q_wdth_mltplr * rc_quan_p->q_wdth_unit ;

		// end : init per block 
	}
}

int
get_some_code_elements( int from, int to )
{
	int ce_type, i ;

	for ( i = from; i <= to; i++ )
	{
		if ( !get_next_type ( &ce_type ))
		{
			fprintf( stderr, "%s: failed from %d to %d\n", __func__, from, to ) ;
			return ( 0 ) ;
		}	

		if ( ce_type != i )
		{
			fprintf( stderr, "%s: want %d got %d \n", __func__, i, ce_type ) ;
			return ( 0 ) ;
		}

		if ( !get_next_element ( ce_type, rc_cep[ i ] ))
		{
			fprintf( stderr, "%s: get_next_element %d failed\n", __func__, ce_type ) ;
			return ( 0 ) ;
		}

		p_element( i, "get_some_code_elements", rc_cep[i] ) ;
	}
	return ( 1 ) ;
}

void
init_solver_beta( struct beta *betap, struct CS_DecParams *opt_p, struct rc_SensingMatrixWH *smhp )
{
	betap->A = opt_p->beta_A0 ; // sens_mtrx.normAtA ??
	betap->D = opt_p->beta_D0 ;

	betap->final.A = opt_p->beta_A ; // sens_mtrx.normAtA ?? Inf ??
	betap->final.D = opt_p->beta_D ; // Inf ??

	betap->scldA = beta.A * ( 1.0 / smhp->sqr_order ) ;
	betap->final.scldA = beta.final.A * ( 1.0 / smhp->sqr_order ) ;

#ifdef CUDA_DBG 
	printf("%s : beta A %f D %f scldA %9.9f final A %f D %f scldA %f order %d\n",
		__func__,
		betap->A,
		betap->D,
		betap->scldA,
		betap->final.A,
		betap->final.D,
		betap->final.scldA,
	    smhp->sqr_order ) ;
#endif
}

/* 
	q_maxerr : q_step / 2 ...
	q_stdv_err : pix std deviation err
*/

void
init_solver_eps ( struct eps *eps_p, int num_of_meas, struct CS_DecParams *opt_p,
	float q_maxerr, float q_stdv_err, int n_sprsvec  )
{

#ifdef CUDA_DBG 
	printf("%s: meas %d max %f stdv %f sprs %d \n", __func__,
		num_of_meas, q_maxerr, q_stdv_err, n_sprsvec ) ;
#endif 

	eps_p->lgrng_chg = ( opt_p->eps_lgrng_chg_init > opt_p->eps_lgrng_chg ) ?
		opt_p->eps_lgrng_chg_init : opt_p->eps_lgrng_chg ;

	eps_p->lgrng_chg_final = opt_p->eps_lgrng_chg ;
	eps_p->lgrng_chg_rate = opt_p->eps_lgrng_chg_rate ;

	eps_p->A_maxerr = q_maxerr + opt_p->eps_A_maxerr * q_stdv_err ;
	eps_p->A_sqrerr = opt_p->eps_A_sqrerr * sqrtf ( num_of_meas * 
		( pow ( q_stdv_err, 2.0 ) + ( pow ( q_maxerr, 2.0 ) / 3.0 ))) ;
	eps_p->D_maxerr = opt_p->eps_D_maxerr ;
	eps_p->D_sqrerr = opt_p->eps_D_sqrerr * sqrtf(( float ) n_sprsvec ) ;

#ifdef CUDA_DBG 
	printf("%s: chg %f final %f rate %f a_max %f a_sqr %f d_max %f d_sqr %f\n",
		__func__,
		eps_p->lgrng_chg ,
		eps_p->lgrng_chg_final ,
		eps_p->lgrng_chg_rate ,
		eps_p->A_maxerr ,
		eps_p->A_sqrerr ,
		eps_p->D_maxerr ,
		eps_p->D_sqrerr ) ;
#endif 
}
