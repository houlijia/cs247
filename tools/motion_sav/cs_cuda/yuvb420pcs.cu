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
#include "cs_whm_encode_b.h"
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
#include "cs_dct.h"

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

// proto

void fix_it ( int fin, int fout ) ;
int allocate_d_mem() ;
int allocate_h_mem() ;
int setup_perm_tbls( char *, char *) ;
int do_measurement ( int fin, int fout ) ;
int make_one_component( int fout,
	char *hin_a, int in_size, 
	int x, int y, int xbdim, int ybdim, 
	int need_interpolate, struct frame_list *fp ) ;

// misc buffer size in elements and size
static int blk_size_e, blk_size_i, perm_size_e, perm_size_i ;
static int perm_blk_size_e ;

// stat
static int total_d_mem = 0 ;

// output file
FILE *md_filep = NULL ;

// host buffers
static char *ybufp = NULL, *vbufp = NULL, *ubufp = NULL ;
static int *outbufp = NULL ;

static int nblk_in_x = 1, nblk_in_y = 1 ;

// for expand u/v to the same size as y
static int do_interpolate = 1 ; // default is yes

// device buffers ... *dperm_lp is one of cube_info[i].cube_perm
int *dperm_rp = NULL, *din_b1 = NULL, *din_b2 = NULL ;
int *dperm_ml_rp = NULL, *dperm_ml_lp = NULL ;	// read the ML file

// 0:inner, 1:side, 2:corner 
static struct cube cube_info[ CUBE_INFO_CNT ] ;
static struct cube wcube_info[ CUBE_INFO_CNT ] ;	// a working copy from cube_info above

static double comp_ratio_f = 100.0 ;

static int do_shift = 0 ;

// for analysis ...

static struct cs_xyz *d_cs_xyzp = NULL ;
static int *d_host_io = NULL ;

// misc

static int inner_cube_size = 0 ;

int cs_config_check( struct cs_config *csp ) ;

// rand

struct RndCState rnd_state_1 ;
struct RndCState rnd_state_2 ;

static int blocks_processed = 0 ; // total number of frame blocks processed ...
static int first_block = 1 ;
static int in_block_to = 0 ;	// next "empty" block, for overlap in T domain
static int *cudadbgp = NULL ;	// 256k entry
static struct cs_config csc ;

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

static const char *timer_name[] = {
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

void
pusage( const char *s )
{
	printf("Usage: %s -f configfilename.json\n", s ) ;
}

main( int ac, char *av[] )
{
	int fin, fout ;
	char opt ;
	// char opt, *finname = NULL, *foutname = NULL ;
	char *configfile = NULL ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	while ((opt = getopt(ac, av, "f:")) != -1) 
	{
		printf(" opt %c \n", opt ) ;

		switch (opt) {
		case 'f' :
			configfile = optarg ;

			break ;
		}
	}

	if ( configfile == NULL )
	{
		pusage( av[0] ) ;
		return ( 1 ) ;
	}

	cs_config_init( &csc ) ;

	if ( !cs_config ( configfile, &csc ))
	{
		pusage( av[0] ) ;
		return ( 2 ) ;
	}

	cs_config_p ( &csc ) ;

	if ( !cs_config_check( &csc ))
	{
		pusage( av[0] ) ;
		return ( 1 ) ;
	}

	comp_ratio_f = (( double ) csc.comp_ratio / 100.0 ) ;

	fprintf( stderr, "x/y (%d, %d) blk x/y/z ( %d, %d, %d) in %s out %s yonly %d "
		"swap %d\n",
		csc.frame_x, csc.frame_y, csc.x_block, csc.y_block, csc.z_block, csc.finname, csc.foutname,
		csc.y_only, csc.do_swap ) ;

	fprintf( stderr, "adj x/y ( %d, %d ) expand x/y/z ( %d, %d, %d ) \n",
		csc.adj_x, csc.adj_y, csc.xadd, csc.yadd, csc.zadd ) ; 

	fprintf( stderr, "weight %d dbg %x\n", csc.weight_scheme, csc.dbg_flag ) ; 

	fprintf( stderr, "cube x/y/z %d %d %d comp %f\n", csc.cubex,
		csc.cubey, csc.cubez, comp_ratio_f ) ; 

	fprintf( stderr, "edge x/y %d %d\n",
		csc.edge_x, csc.edge_y ) ;

	if ( csc.overlap_z )
		fprintf( stderr, "overlap %d\n", csc.overlap_z ) ;

	fprintf( stderr, "perm %d %s\n", csc.do_permutation, csc.permdir ) ;

	if (strlen(csc.finname))
	{
		fin = open( csc.finname, O_RDONLY ) ;

		if ( fin == -1 )
		{
			printf("file %s does not exist\n", av[1]) ;
			exit( 1 ) ;
		}
	}

	fout = open( csc.foutname, O_CREAT | O_TRUNC | O_WRONLY, S_IRWXU ) ;

	if ( fout == -1 )
	{
		printf("file %s open failed %d\n", csc.foutname, errno ) ;
		exit( 1 ) ;
	}

	dbg_init ( 256 * 1024 * 1024 * sizeof ( 4 )) ;

	if ( !allocate_d_mem())
	{
		printf("%s: d_mem allocation failed\n", __func__ ) ;
		exit( 1 ) ;
	}

	if (( cudadbgp = dbg_d_malloc_i ( 1024 * 256 )) == NULL )
	{
		exit( 1 ) ;
	}

	clear_device_mem_i( cudadbgp, 1024 * 256 ) ;

	if ( !allocate_h_mem())
	{
		printf("%s: h_mem allocation failed\n", __func__ ) ;
		exit( 1 ) ;
	}

#if __BYTE_ORDER__ == __BIG_ENDIAN
	opt = CS_CO_BIGENDIAN ;
#else
	opt = 0 ;
#endif

	if ( strlen( csc.ipcam_string))
	{
		if ( !cs_ipcam_init ( csc.z_block, csc.frame_x, csc.frame_y, csc.ipcam_string,
			nblk_in_x, nblk_in_y, csc.md_x, csc.md_y, csc.disp_th_x, csc.disp_th_y ))
		{
			printf("ipcam_init failed\n") ;
			exit( 1 ) ;
		}
	}

	if ( opt ) 	// local machine is big endian
	{
		if ( csc.do_swap )
			opt = 0 ; // back to little endian
	} else
	{
		if ( csc.do_swap )
			opt = CS_CO_BIGENDIAN ;
	}

	if ( csc.do_permutation )
		opt |= ML_PERM ;

	if ( csc.do_cube )
		opt |= DOUBLE_PERM ;

	fprintf( stderr, "%s: size of header %d\n", __func__,
		sizeof ( struct cs_header )) ;

	// + 2 ... 1 is for the center of the edge detection rectangle
	// the other 1 is for the shift/move to make sense
	if (( cube_info[0].x <= (( csc.edge_x << 1 ) + 2 )) ||
		( cube_info[1].x <= (( csc.edge_x << 1 ) + 2 )) ||
		( cube_info[2].x <= (( csc.edge_x << 1 ) + 2 )) ||
		( cube_info[0].y <= (( csc.edge_y << 1 ) + 2 )) ||
		( cube_info[1].y <= (( csc.edge_y << 1 ) + 2 )) ||
		( cube_info[2].y <= (( csc.edge_y << 1 ) + 2 )))
	{
		fprintf( stderr, "%s: error cube x %d %d %d edge x %d cube y %d %d %d edge y %d\n",
			__func__,
			cube_info[0].x,
			cube_info[1].x,
			cube_info[2].x,
			csc.edge_x, 
			cube_info[0].y,
			cube_info[1].y,
			cube_info[2].y,
			csc.edge_y ) ;

		exit( 23 ) ;
	}

	if ( !cs_put_header ( fout, CS_CD_YUV420P, 
		(( csc.y_only )? Y_COMP_ONLY : 0 ) | opt,
		WALSH_HADAMARD_MATRIX,
		csc.frame_x, csc.frame_y,
		csc.x_block, csc.y_block, csc.z_block,
		cube_info[0].x, cube_info[0].y, cube_info[0].z, 
		cube_info[1].x, cube_info[1].y, cube_info[1].z, 
		cube_info[2].x, cube_info[2].y, cube_info[2].z, 
		csc.overlap_x, csc.overlap_y, csc.overlap_z,
		csc.xadd, csc.yadd, csc.zadd,
		csc.adj_x, csc.adj_y,
		csc.edge_x, csc.edge_y,
		csc.md_x, csc.md_y, csc.md_z,
		csc.weight_scheme ))
	{
		printf("can't write header\n") ;
		exit( 1 ) ;
	} 

	fprintf( stderr, "%s: do_swap %d do_interpolate %d do_permutation %d do_cube %d "
		"do_comp_ratio %d do_block %d do_analysis %d do_one %d do_not_seek %d \n", 
		__func__, 
		csc.do_swap,
		csc.do_interpolate,
		csc.do_permutation,
		csc.do_cube,
		csc.do_comp_ratio,
		csc.do_block, 
		csc.do_analysis, 
		csc.do_one, 
		csc.do_not_seek ) ; 

	if ( md_filep )
		ma_report_header ( md_filep, csc.frame_y, csc.frame_x, 0, 1, 2, 1 ) ;

	if ( !do_measurement( fin, fout ))
	{
		printf("do_measurement: failed\n") ;
	    exit( 1 ) ;
	}	

	close ( fin ) ;
	close ( fout ) ;

	if ( md_filep )
		fclose ( md_filep ) ;
}

void
do_tst_longlong()
{
	fprintf(stderr, "%s: size of ll %d\n", __func__, sizeof ( long long )) ;

	h_tst_longlong (( long long *)din_b2, 100 ) ;
	dbg_p_d_data_ll ("tst long long",( long long *)din_b2,
		100 * sizeof ( long long )) ;
}

// allocate the d mem ... and init the perm tables
int
allocate_d_mem()
{
	int ana_size, ocube_size = 0, cube_size, *hdp = NULL, nmea, nz,
		i, j, k, xx, yy, zz, x, y, z ;
		
	if ( csc.do_cube )
	{
		if (( k = cudaMalloc( &d_cs_xyzp, sizeof ( *d_cs_xyzp ) *
			CUBE_INFO_CNT + sizeof ( int ) * 10 )) != cudaSuccess )
		{
			printf("%s: cs_xyzp alloc failed %d\n", __func__, k ) ;
			return ( 0 ) ;
		}

		total_d_mem += sizeof ( *d_cs_xyzp ) * CUBE_INFO_CNT + sizeof ( int ) * 10 ;

		d_host_io = ( int * )( d_cs_xyzp + CUBE_INFO_CNT ) ;

		fprintf( stderr, "%s: d_cs_xyzp %p d_host_io %p\n",
			__func__, d_cs_xyzp, d_host_io ) ;

		nmea = ( int )(( double )( csc.x_block * csc.y_block * csc.z_block ) *
			comp_ratio_f ) ;

		xx = csc.x_block + csc.adj_x ;
		yy = csc.y_block + csc.adj_y ;
		zz = csc.z_block + csc.zadd ;

		ana_size = 0 ;

		for ( i = 0 ; i <  CUBE_INFO_CNT  ; i++ )
		{
			x = csc.cubex ;
			y = csc.cubey ;
			z = csc.cubez ;

#ifdef CUDA_DBG 
			printf("%s:i %d mea %d -- xx %d %d %d x %d %d %d\n",
				__func__, i, nmea, xx, yy, zz, x, y, z ) ; 
#endif 

			// inside is 1, side is 1/2, corner is 1/4
			j = ( int  )pow((double)2,(double)i) ;
			k = nmea / j ;

			printf("%s: i %d nmea %d k %d j %d comp %f\n", __func__, i,
				nmea, k, j, comp_ratio_f ) ;

			// the "+ 2" is to make sure at least the x and y has 2x2 block to cmp
			if ( h_do_find_perm_size ( xx, yy, zz, &x, &y, &z,
				csc.z_block, k, ( csc.edge_x + csc.md_x ) * 2 + 2,
				( csc.edge_y + csc.md_y ) * 2 + 2) == 0 )
			{
				printf( "%s: cube 1 failed \n", __func__ ) ;
				return ( 0 ) ;
			}

			cube_size = x * y * z * sizeof ( int ) ;

			if (( k = cudaMalloc( &cube_info[i].dp, cube_size )) != cudaSuccess )
			{
				printf("%s: cube alloc failed %d %d %d \n", __func__, i,
					cube_size, k ) ;
				return ( 0 ) ;
			}

			total_d_mem += cube_size ;

			if ( hdp == NULL )
			{
				ocube_size = cube_size ;
				hdp = ( int * )malloc( cube_size + 10 ) ;
				if ( hdp == NULL )
				{
					printf("%s: cube host alloc failed %d %d \n",
						__func__, i, cube_size ) ;
					return ( 0 ) ;
				}

			}

			cube_info[i].x = x ;
			cube_info[i].y = y ;
			cube_info[i].z = z ;
			cube_info[i].size = x * y * z ;

			printf("%s: i %d x/y/z %d %d %d\n", __func__, i, x, y, z );

			cube_info[i].cube_perm = NULL ;

			if (( x * y * z * sizeof ( int )) > ocube_size )
			{
				printf("%s: cube size error %d %d %d %d \n",
					__func__, x, y, z, ocube_size ) ;
				return ( 0 ) ;
			}

			h_do_get_perm_matrix( hdp, xx, yy, zz, x, y, z, &cube_info[i].sink ) ;
			
			if (( k = cudaMemcpy( cube_info[i].dp, hdp, cube_size,
				cudaMemcpyHostToDevice)) != cudaSuccess )
			{
				printf("%s:cube download fail: loop %d %d\n", __func__, i, k ) ;
				return ( 0 ) ;
			}

			x = cube_info[i]. x - ( csc.edge_x << 1 ) - ( csc.md_x << 1 ) ;
			y = cube_info[i]. y - ( csc.edge_y << 1 ) - ( csc.md_y << 1 ) ;

			if (( x <= 0 ) || ( y <= 0 )) 
			{
				printf("%s: i %d x/y %d %d cube %d %d edge %d %d motion detect %d %d \n",
					__func__, i, x, y,
					cube_info[i].x, cube_info[i].y, 
					csc.edge_x, csc.edge_y, 
					csc.md_x, csc.md_y ) ;
				return ( 0 ) ;
			}

			k = ( x * y * ( cube_info[i].z  - csc.md_z + 1 )  + NUM_OF_HVT_INDEX ) *
				((( csc.md_x << 1 ) + 1 ) * (( csc.md_y << 1 ) + 1) * ( csc.md_z - 1 ) + 1 )  ;

			printf("%s: x %d y %d k %d ana_size %d\n", __func__, x, y, k, ana_size ) ;
			printf("%s: md_x/y/z %d %d %d cube.z %d\n", __func__, csc.md_x, csc.md_y, csc.md_z,
				cube_info[i].z ) ;

			if ( ana_size < k )
				ana_size = k ;

#ifdef CUDA_OBS 
			dbg_p_d_data_i ( "cube tbl", cube_info[i].dp, 
				cube_info[i].size ) ;
#endif 
		}

		// set the config
		h_set_config( d_cs_xyzp, cube_info ) ;

#ifdef CUDA_DBG 
		dbg_p_d_data_i ( "config", ( int *)d_cs_xyzp, 3 * CUBE_INFO_CNT ) ;
#endif 

		printf("%s: ana_size %d\n", __func__, ana_size ) ;

		free ( hdp ) ;
	}

	if ( csc.do_reconstruction )
	{
		if ( !h_do_dct_init())
		{
		  printf("%s: h_do_dct_init failed \n", __FILE__) ;
			return ( 0 ) ;
		}
	}

	x = csc.frame_x + ( csc.xadd << 1 ) ;
	y = csc.frame_y + ( csc.yadd << 1 ) ;

	nblk_in_x = ( x - csc.overlap_x ) / ( csc.x_block - csc.overlap_x )  ;
	j = ( x - csc.overlap_x ) % ( csc.x_block - csc.overlap_x )  ;

	if ( j )
	{
		fprintf( stderr, "%s: x %d x_block %d overlap_x %d nblk_in_x %d j %d\n",
			__func__, x, csc.x_block, csc.overlap_x, nblk_in_x , j ) ;
		return ( 0 ) ;
	}

	nblk_in_y = ( y - csc.overlap_y ) / ( csc.y_block - csc.overlap_y )  ;
	j = ( y - csc.overlap_y ) % ( csc.y_block - csc.overlap_y )  ;

	if ( j )
	{
		fprintf( stderr, "%s: y %d y_block %d overlap_y %d nblk_in_y %d j %d\n",
			__func__, y, csc.y_block, csc.overlap_y, nblk_in_y, j ) ;
		return ( 0 ) ;
	}

	if (( nblk_in_x < 2 ) || ( nblk_in_y < 2 ))
	{
		fprintf( stderr, "%s: not enough blks nblk_x %d nblk_y %d x/y %d %d ox/y %d %d\n",
			__func__, nblk_in_x, nblk_in_y, x, y, csc.overlap_x, csc.overlap_y ) ;
		return ( 0 ) ;
	}

	nz = csc.z_block + csc.zadd ;

	blk_size_e = ( csc.x_block + csc.adj_x ) * ( csc.y_block + csc.adj_y ) * nz ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: blk_size %d nblk_in_x %d nblk_in_y %d nz %d x_b %d"
		" y_b %d ax %d ay %d\n",
		__func__, blk_size_e, nblk_in_x, nblk_in_y, nz, csc.x_block,
		csc.y_block, csc.adj_x, csc.adj_y ) ;
#endif 

	if ( csc.do_permutation )
		blk_size_e++ ;

	perm_size_e = max_log2 ( blk_size_e ) ;

	if ( csc.do_analysis )
		ana_size = ( ana_size > perm_size_e ) ? ana_size : perm_size_e ;
	else
		ana_size = perm_size_e ;

	blk_size_e = ana_size * nblk_in_x * nblk_in_y ;
	perm_blk_size_e = perm_size_e * nblk_in_x * nblk_in_y ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: do_perm %d perm_size %d blk_size %d ana_size %d perm_blk_size %d\n", 
		__func__, csc.do_permutation, perm_size_e, blk_size_e, ana_size, perm_blk_size_e ) ;
#endif 

	do_shift = weight_sft( csc.weight_scheme, perm_size_e, csc.x_block, csc.y_block ) ;

	blk_size_i = sizeof ( int ) * blk_size_e ;
	perm_size_i = sizeof ( int ) * perm_size_e ;

	if (( i = cudaMalloc( &din_b1, blk_size_i )) != cudaSuccess )
	{
		printf("%s: din_b1 failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	total_d_mem += blk_size_i ;

	if (( i = cudaMalloc( &din_b2, blk_size_i )) != cudaSuccess )
	{
		printf("%s: din_b2 failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	total_d_mem += blk_size_i ;

	if (( i = cudaMalloc( &dperm_ml_lp, perm_size_i )) != cudaSuccess )
	{
		printf("%s: dperm_ml_lp failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	total_d_mem += perm_size_i ;

	for ( k = 0 ; k <  CUBE_INFO_CNT  ; k++ )
	{
		if (( i = cudaMalloc( &cube_info[k].cube_perm,
			perm_size_i )) != cudaSuccess )
		{
			printf("%s: dperm_cube failed %d %d \n", __func__, k, i ) ;
			return ( 0 ) ;
		}
		total_d_mem += perm_size_i ;
	}

	if (( i = cudaMalloc( &dperm_ml_rp, perm_size_i )) != cudaSuccess )
	{
		printf("%s: dperm_ml_rp failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	total_d_mem += perm_size_i ;

	if (( i = cudaMalloc( &dperm_rp, perm_size_i )) != cudaSuccess )
	{
		printf("%s: dperm_rp failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	total_d_mem += perm_size_i ;

	printf("%s: DEV GOOD  == b1 %p b2 %p bsize %d \n"
		"	pR %p psize %d mllp %p mlrp %p total_d_mem %d\n",
		__func__, din_b1, din_b2, blk_size_i,
		dperm_rp, perm_size_i, dperm_ml_lp, dperm_ml_rp,
		total_d_mem ) ;

	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		fprintf( stderr, "%s: dp %p xyz %d %d %d size %d sk %d perm %p\n",
			__func__,
			cube_info[i].dp,
			cube_info[i].x,
			cube_info[i].y,
			cube_info[i].z,
			cube_info[i].size,
			cube_info[i].sink,
			cube_info[i].cube_perm
			) ;
	}

	i = ( int )( log2(( double ) perm_size_e )) ;

	if ( !permutation_load( i, csc.permdir, dperm_ml_lp,
		dperm_ml_rp ))
	{
		printf("%s: y perm load failed\n", __func__ ) ;
		return ( 0 ) ;
	}

	return ( 1 ) ;
}

/* 
allocate_h_mem: allocate the buffers for used in memory.

need: frame_x, frame_y, z_block, blk_size_i, y_only
*/
int 
allocate_h_mem()
{
	int ysize, uvsize ;

	// for YUV420

	ysize = csc.frame_x * csc.frame_y * csc.z_block ;
	uvsize = ysize >> 2 ;	

	if ( !strlen(csc.ipcam_string))
	{
		ybufp = ( char * ) malloc ( ysize ) ;
	}

	if ( !csc.y_only )
	{
		ubufp = ( char * ) malloc ( uvsize ) ;
		vbufp = ( char * ) malloc ( uvsize ) ;
	} else
	{
		ubufp = vbufp = NULL ;
	}

	if (( !ybufp && ( !strlen(csc.ipcam_string))) || ( !csc.y_only && ( !ubufp || !vbufp )))
	{
		printf("%s: 1 malloc failed \n", __func__ ) ;
		return ( 0 ) ;
	}

	outbufp = ( int * ) malloc ( blk_size_i ) ;

	if ( outbufp == NULL )
	{
		printf("%s: 2 malloc failed \n", __func__ ) ;
		return ( 0 ) ;
	}

	printf("%s: HOST GOOD ysize %d uv %d ybufp %p\n"
		"	 u %p v %p o %p blksize %d\n",
		__func__, ysize, uvsize, ybufp, ubufp, vbufp, outbufp, blk_size_i ) ; 
	return ( 1 ) ;
}

int
setup_perm_tbls( int *d_bp )
{
	int i ;
	unsigned int ran ;
	RndC_uint32 ran1 ;

	// assume the folloing
	// 1. left selection tbls are in cube_info[i].cube_perm
	// 2. ml seq perm tbl is in perm_ml_lp and perm_ml_rp ;
	// 3. dperm_rp will be updated ...

	// right perm first ...

	// ran = rand() ;

	randi_RndC( &rnd_state_1, RndC_uint32( perm_size_e ), (size_t)1, &ran1 ) ;

	ran = ran1 - 1 ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: right ran %d perm_size %d \n", 
		__func__, ran, perm_size_e ) ;
#endif 

	h_do_perm_selection_R ( d_bp, perm_size_e, ran ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("right selection", d_bp, perm_size_e) ;
#endif 

	h_do_permutation_double ( d_bp, dperm_ml_rp, dperm_rp, perm_size_e ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("right merge", dperm_rp, perm_size_e ) ;
#endif 

	// now the left 

	randi_RndC( &rnd_state_2, RndC_uint32( perm_size_e ), (unsigned int)1, &ran1 ) ; 

	ran = ran1 - 1 ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: left ran %d perm_size %d \n", 
		__func__, ran, perm_size_e ) ;
#endif 

	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		h_do_perm_selection_L ( d_bp, perm_size_e, cube_info[i].dp,
			cube_info[i].size, ran, cube_info[i].sink ) ;

		h_do_permutation_double ( d_bp, dperm_ml_lp,
			cube_info[i].cube_perm, perm_size_e ) ;	// orig one
	}

	return ( 1 ) ;

}

int
get_one_chunk( int fin, char *bp, int size )
{
	int i ;

	if (( i = read ( fin, bp, size )) < 0 )
	{
		printf("%s: y read failed errno %d\n", __func__,
			errno ) ;
		return ( 0 ) ;
	}

	if ( !i )
		return ( 2 ) ;

	if ( i != size )
	{
		printf("%s: read failed i %d\n", __func__, i ) ;
		return ( 0 ) ;
	}

	return ( 1 ) ;
}

int
get_block_input ( int fin )
{
	int offsety, offsetuv, sizey, sizeuv, i ;
	int cnt ;

	offsety = 0 ;
	offsetuv = 0 ;
	sizey = csc.frame_x * csc.frame_y ;
	sizeuv = sizey >> 2 ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: fin %d x %d y %d zb %d yin %x uin %x vin %x\n",
		__func__, fin, csc.frame_x, csc.frame_y, csc.z_block, ybufp, ubufp, vbufp ) ;
#endif 

	cnt = csc.z_block ;

	if ( !first_block )
		cnt -= csc.overlap_z ;

	while ( cnt-- )
	{
		offsety = in_block_to * sizey ;
		i = get_one_chunk( fin, &ybufp[ offsety ], sizey ) ;

		if ( i != 1 )
			return ( i ) ;

		if ( csc.y_only )
		{
			if ( !csc.do_not_seek )
			{
				if (( i = lseek ( fin, sizeuv << 1, SEEK_CUR )) < 0 )
				{
					printf("%s: failed lseek %d\n",
						__func__, errno ) ;
					return ( 0 ) ;
				}
			}
		} else
		{
			offsetuv = in_block_to * sizeuv ;
			i = get_one_chunk( fin, &ubufp[ offsetuv ], sizeuv ) ;

			if ( i != 1 )
				return ( i ) ;
	
			i = get_one_chunk( fin, &vbufp[ offsetuv ], sizeuv ) ;
	
			if ( i != 1 )
				return ( i ) ;
		}

		in_block_to++ ;

		if ( in_block_to == csc.z_block )
			in_block_to = 0 ;
		
	}
#ifdef CUDA_OBS 
	dbg_pdata_c ( "get_block_input: ybuf", ybufp, sizey * z_block ) ;

	if ( !y_only )
	{
		dbg_pdata_c ( "get_block_input: ubuf", ubufp, sizeuv * z_block ) ;
		dbg_pdata_c ( "get_block_input: vbuf", vbufp, sizeuv * z_block ) ;
	}
#endif 
	return ( 1 ) ;
}

int
do_measurement ( int fin, int fout )
{
	int cnt, i, frame_cnt ;
	double d, ad, aut, ast ;
	clock_t tut, tst ;
	struct frame_list *fp ;
	static RndC_uint32 rand_key1 = 0 ;
	static RndC_uint32 rand_key2 = 0 ;

	// d_in_p2 will be used in the fast transform ... 
	// d_in_p has data before blocking ...

	omp_timer_init( CS_TIMER_COUNT ) ;
	cs_timer_init( 1 ) ;

	if ( strlen(csc.ipcam_string))
		cs_ipcam_start() ;

	frame_cnt = 0 ;
	first_block = 1 ;
	while ( 1 )
	{
		if ( csc.do_permutation )
		{
			init_RndC( &rnd_state_1, rand_key1 ) ;
			init_RndC( &rnd_state_2, rand_key2 ) ;

			setup_perm_tbls( din_b1 ) ;

			rand_key1++ ;
			rand_key2++ ;

			// printf("randkey1 %d randkey2 %d\n", rand_key1, rand_key2 ) ;
		}

		if ( csc.do_one-- )
		{
			if (strlen( csc.ipcam_string) == 0 )
			{
				i = get_block_input ( fin ) ;
				fp = NULL ;
			} else
			{
#ifdef CUDA_DBG
				printf("do_measurement: OUT ===\n") ;
#endif 
				fp = cs_ipcam_get() ;
#ifdef CUDA_DBG 
				printf("do_measurement: IN ===\n") ;
#endif 
				ybufp = fp->gbp ;
				i = 1 ;
			}
		} else
			i = 2 ; // for debug only do once 

		// dbg_pdata_c ( "after get block input", ybufp, frame_x * frame_y * zb ) ;

		if ( !i )
		{
			printf("%s:failed %d\n", __func__, frame_cnt ) ;
			return ( 0 ) ;
		}

		if ( i == 2 )
		{
			printf("\n%s: frame_cnt %d size %d\n",
				__func__, frame_cnt, frame_cnt * csc.frame_x * csc.frame_y ) ;

			printf("=== counters below are based on the block cnt\n") ;

			for ( i = 0 ; i < CS_TIMER_COUNT ; i++ )
			{
				omp_timer_get ( i, &d, &cnt, &ad ) ;
				printf("%s ::: %f cnt %d average %f ms\n", 
					timer_name[i], d, cnt, ad ) ;
			}

			cs_timer_get ( 0, &tst, &tut, &cnt, &ast, &aut ) ;
			printf("overall st %d ut %d cnt %d ast %f aut %f == total %f ms\n",
				tst, tut, cnt, ast, aut, ast + aut ) ; 
			return ( 1 ) ;
		}

		memcpy( wcube_info, cube_info, sizeof ( cube_info )) ;
		h_set_config ( d_cs_xyzp, wcube_info ) ;

		// do y
	
		i = make_one_component( fout, ybufp,
			csc.frame_x * csc.frame_y * csc.z_block, 
			csc.frame_x, csc.frame_y,
			csc.x_block, csc.y_block, 
			0, fp ) ;

		if ( !i )
		{
			printf("%s:failed frame_cnt %d\n", __func__, frame_cnt ) ;
			return ( 0 ) ;
		}

		if ( !csc.y_only )
		{
			// should revisit din_size_u ... due to the expand
		// do y
	
			if ( !make_one_component( fout, ubufp,
				csc.frame_x * csc.frame_y * csc.z_block >> 2 , csc.frame_x >> 1, csc.frame_y >> 1,
				csc.x_block >> 1, csc.y_block >> 1, 1, fp ))
			{
				printf("%s:failed i %d frame %d\n", __func__, i, frame_cnt ) ;
				return ( 0 ) ;
			}

			if ( !make_one_component( fout, vbufp,
				csc.frame_x * csc.frame_y * csc.z_block >> 2, csc.frame_x >> 1, csc.frame_y >> 1,
				csc.x_block >> 1, csc.y_block >> 1, 1, fp ))
			{
				printf("%s:failed i %d frame %d\n", __func__, i, frame_cnt ) ;
				return ( 0 ) ;
			}

		}

		frame_cnt += ( csc.z_block - csc.overlap_z ) ;

		first_block = 0 ;

	}
}

int *
the_other_d_buf ( int *p )
{
  if ( p == din_b1 )
    return ( din_b2 ) ;
  return ( din_b1 ) ;
}
 
/*
fout 	  : out file descriptor
db_1p     : add of input buffer on device
dout_a    : add of output buffer on device
hin_a     : add of host buffer for input from file
ysize     : size of input for this comp in byte // real dimension side

dout_size : size of output buffer on device in byte // log2 size
din_size  : size of input buffer on device in byte // log2 size

din_a2 : add of input buffer on device before block // NULL if block is not needed
x, y		: frame dimension
xbdim, ybdim : block x/y dimension
blk_dst_size : log2 of x/y/z block size 
d_out_p2:	add of output buffer on device for do_perm
fp:	frame_list pointer ... for ipcam 
*/

int
make_one_component( int fout,
	char *hin_a, int in_size, 
	int x, int y, int xbdim, int ybdim, 
	int need_interpolate, struct frame_list *fp )
{
	int *outp, *d_currp, *d_nextp ;
	char *d_curr_cp ;
	int orig, hvt_size, overall_size, i, rec_size, j ;
	int k, frame_size = x * y ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: inbuf %p in_size %d x/y %d %d blk x/y %d %d\n"
		"interpo %d\n",
		__func__, hin_a, in_size, x, y, xbdim, ybdim, need_interpolate ) ;
#endif  

	d_currp = din_b1 ;

	// clear_device_mem_c( d_currp, din_size ) ;

	d_nextp = din_b2 ;

	// clear_device_mem_c( d_nextp, din_size ) ;

	omp_timer_on( CS_TIMER_TOTAL ) ;
	cs_timer_on( 0 ) ;

	// copy the frame data ( could be y, u, v ) from host to device
	// take care of z_overlap

	d_curr_cp = ( char * ) d_currp ;
	omp_timer_on( CS_TIMER_MEMCPY_DOWN ) ;

	if (( i = cudaMemcpy( d_currp, hin_a + in_block_to * frame_size,
		( csc.z_block - in_block_to ) * frame_size,
		cudaMemcpyHostToDevice)) != cudaSuccess )
	{
		printf("%s:download fail: first %d\n", __func__, i ) ;
		return ( 0 ) ;
	}

	if ( in_block_to )
	{
		if (( i = cudaMemcpy( d_curr_cp +
			( csc.z_block - in_block_to ) * frame_size, 
			hin_a, in_block_to * frame_size,
			cudaMemcpyHostToDevice)) != cudaSuccess )
		{
			printf("%s:download fail: second %d\n", __func__, i ) ;
			return ( 0 ) ;
		}
	}
	
	omp_timer_off( CS_TIMER_MEMCPY_DOWN ) ;

	if ( csc.dbg_flag & DBG_CP_DOWN )
		dbg_p_d_data_c_mn ( "after memcpy cp in", ( char * )d_currp, in_size, 
			x, y, x ) ; 

	d_nextp = the_other_d_buf ( d_currp ) ;
	
	omp_timer_on( CS_TIMER_C_TO_I ) ;

	h_expand_c_to_i (( char * ) d_currp, d_nextp, in_size ) ; 

	omp_timer_off( CS_TIMER_C_TO_I ) ;

	d_currp = d_nextp ;
	
	if ( csc.dbg_flag & DBG_C_2_I )
		dbg_p_d_data_i_mn ( "after c to i", d_currp, in_size, 
			x, y, x ) ; 

	// to bring u/v component to same size as y

	if ( do_interpolate && need_interpolate )
	{
		exit( 34 ) ;	// LDL QQQ

		d_nextp = the_other_d_buf ( d_currp ) ;

		// might need to adjust for the interpolation,
		// like blk_size_i ... which is based on 'y'
		// and used in h_make_block ... LDL

		omp_timer_on ( CS_TIMER_INTER ) ;

		if ( !h_make_interpolate ( d_currp, d_nextp,
			x, y, csc.z_block, INT_YUV420 ))
		{
			printf("%s: make_interpolate failed\n", __func__ ) ;
			return ( 0 ) ;
		}

		omp_timer_off ( CS_TIMER_INTER ) ;

		d_currp = d_nextp ;

		if ( csc.dbg_flag & DBG_INTER )
			dbg_p_d_data_c ( "after interpolate",( char *)d_currp, in_size ) ;

		xbdim <<= 1 ;
		ybdim <<= 1 ;
		x <<= 1 ;
		y <<= 1 ;

		in_size *= 4 ;
	}

	// expand the data 

	d_nextp = the_other_d_buf ( d_currp ) ;

	omp_timer_on( CS_TIMER_EXPANSION ) ;

	h_expand_frame ( d_currp, d_nextp,
		x, y, csc.xadd, csc.yadd, csc.zadd,
		csc.z_block ) ;

	omp_timer_off( CS_TIMER_EXPANSION ) ;

	d_currp = d_nextp ;

	frame_size = ( x + ( csc.xadd << 1 )) *
		( y + ( csc.yadd << 1 )) ;

	if ( csc.dbg_flag & DBG_EXPAND )
	{
		i = frame_size * ( csc.z_block + csc.zadd ) ;
		dbg_p_d_data_i_mn ( "after expand", d_currp, i, 
			x + ( csc.xadd << 1 ),
			y + ( csc.yadd << 1 ), 
			x + ( csc.xadd << 1 )) ;
	}

	// do the blocking ... move data/append 0/weight

	if ( csc.do_block )
	{
		d_nextp = the_other_d_buf ( d_currp ) ;

		omp_timer_on( CS_TIMER_BLOCKING ) ;

		set_device_mem_i ( d_nextp, perm_blk_size_e, 0 ) ; // this is needed ... 
			// to set the adj_x/adj_y to 0

		h_make_block( d_currp, d_nextp, 
			x + ( csc.xadd << 1 ), y + ( csc.yadd << 1 ), frame_size,
			xbdim, ybdim, csc.z_block, perm_size_e,
			csc.do_permutation, xbdim - csc.overlap_x, ybdim - csc.overlap_y,
			nblk_in_x, nblk_in_y, csc.adj_x, csc.adj_y, 
			csc.weight_scheme, do_shift ) ; 

		omp_timer_off( CS_TIMER_BLOCKING ) ;

		d_currp = d_nextp ;

		if ( csc.dbg_flag & DBG_BLKING )
			dbg_p_d_data_i_mn_skip ( "after blking",
				csc.do_permutation ? d_currp + 1 : d_currp, 
				perm_blk_size_e, 
				xbdim + csc.adj_x, ybdim + csc.adj_y, csc.z_block + csc.zadd,
				xbdim + csc.adj_x, perm_size_e ) ;
#ifdef CUDA_OBS 
			dbg_p_d_data_i_mn ( "after blking",
				do_permutation ? d_currp + 1 : d_currp, 
				perm_blk_size_e, 
				xbdim + adj_x, ybdim + adj_y, xbdim + adj_x ) ;
#endif 

		// NOTE: if do_permutation, then take out the first element
	}

	// do the R permutation here ...

	if ( csc.do_permutation )
	{
		d_nextp = the_other_d_buf ( d_currp ) ;

#ifdef CUDA_OBS 
		{
			int *dp, i1 ;

			dp = outbufp ; 
			for ( i1= 0 ; i1 < perm_size_e ; i1++ )
				*dp++ = i1 ;

			if (( i1 = cudaMemcpy( d_currp, outbufp, perm_size_e * sizeof ( int ) ,
				cudaMemcpyHostToDevice)) != cudaSuccess )
			{
				printf("%s:test cpy fail: %d \n", __func__, i1 ) ;
				return ( 0 ) ;
			}
			dbg_p_d_data_i("R before ", d_currp, perm_size_e) ;
		}
#endif 

#ifdef CUDA_OBS 
		dbg_p_d_data_i("R before ", d_currp + perm_size_e, perm_size_e) ;
#endif 

		omp_timer_on( CS_TIMER_PERMR ) ;

		// input is din_a ... 
		h_do_permutation_R ( d_currp, d_nextp, dperm_rp, perm_blk_size_e,
			perm_size_e ) ;
		
		omp_timer_off( CS_TIMER_PERMR ) ;

		d_currp = d_nextp ;

#ifdef CUDA_OBS 
		dbg_p_d_data_i("R after ", d_currp + perm_size_e, perm_size_e) ;
#endif 

		if ( csc.dbg_flag & DBG_PERM_R )
			dbg_p_d_data_i_mn_skip ( "after perm-R",
				csc.do_permutation ? d_currp + 1 : d_currp, 
				perm_blk_size_e, 
				xbdim + csc.adj_x, ybdim + csc.adj_y, csc.z_block + csc.zadd,
				xbdim + csc.adj_x, perm_size_e ) ;
	}

	// do the transformation ...

	if ( csc.dbg_flag & DBG_WHM )
		dbg_p_d_data_i_mn_skip ( "before whm",
			csc.do_permutation ? d_currp + 1 : d_currp, 
			perm_blk_size_e, 
			xbdim + csc.adj_x, ybdim + csc.adj_y, csc.z_block + csc.zadd,
			xbdim + csc.adj_x, perm_size_e ) ;


	omp_timer_on( CS_TIMER_MEA ) ;

	cs_whm_measurement_b( d_currp, perm_blk_size_e, perm_size_e ) ; 

	omp_timer_off( CS_TIMER_MEA ) ;

	if ( csc.dbg_flag & DBG_WHM )
		dbg_p_d_data_i_mn_skip ( "after whm",
			csc.do_permutation ? d_currp + 1 : d_currp, 
			perm_blk_size_e, 
			xbdim + csc.adj_x, ybdim + csc.adj_y, csc.z_block + csc.zadd,
			xbdim + csc.adj_x, perm_size_e ) ;

	// do the L permutation here ...

	if ( csc.do_permutation )
	{
		d_nextp = the_other_d_buf ( d_currp ) ;

		// input is din_a ... 

		omp_timer_on( CS_TIMER_PERML ) ;

		h_do_permutation_Lv2 ( d_currp, d_nextp,
			wcube_info[0].cube_perm,
			wcube_info[1].cube_perm,
			wcube_info[2].cube_perm,
			perm_blk_size_e,
			perm_size_e,
			nblk_in_x,
			nblk_in_y ) ;
		
		omp_timer_off( CS_TIMER_PERML ) ;

		d_currp = d_nextp ;

		if ( csc.dbg_flag & DBG_PERM_L )
			dbg_p_d_data_i_mn_skip ( "after L-perm",
				csc.do_permutation ? d_currp + 1 : d_currp, 
				perm_blk_size_e, 
				xbdim + csc.adj_x, ybdim + csc.adj_y, csc.z_block + csc.zadd,
				xbdim + csc.adj_x, perm_size_e ) ;
			// dbg_p_d_data_i ( "after l perm", d_currp, perm_blk_size_e ) ;
	}

	// copy_vec ... the to-size is the "inner block" size

	d_nextp = the_other_d_buf ( d_currp ) ;

	inner_cube_size = wcube_info[0].size ;
	overall_size = inner_cube_size * nblk_in_x * nblk_in_y ;

#ifdef CUDA_DBG 
	set_device_mem_i( d_nextp, perm_blk_size_e, 101 ) ;
#endif

	if ( !( k = h_do_copy_vec ( d_currp, d_nextp,
		overall_size, 
		perm_size_e,
		inner_cube_size )))
	{
		printf("%s: copy_vec failed %d i %d \n", __func__, k, i ) ;
		return ( 0 ) ;
	} 
	
	d_currp = d_nextp ;

	if ( csc.dbg_flag & DBG_COPY_DONE )
	{
		dbg_p_d_data_i_mn_v2 ( "copy after", d_currp, overall_size, wcube_info[0].x,
			wcube_info, nblk_in_x, nblk_in_y ) ;
	}

	blocks_processed++ ;

	// from here and on ... only the L-selected measurements

	if ( csc.do_analysis )
	{
		// edge detection 

		omp_timer_on ( CS_TIMER_ANALYSIS ) ;

		d_nextp = the_other_d_buf ( d_currp ) ;

		omp_timer_on ( CS_TIMER_ANALYSIS_EDGE ) ;

		h_do_edge_detection_v2 ( d_currp, d_nextp, overall_size,
			d_cs_xyzp, csc.edge_x, csc.edge_y, nblk_in_x, nblk_in_y,
			cube_info) ;

		omp_timer_off ( CS_TIMER_ANALYSIS_EDGE ) ;

		if ( csc.dbg_flag & DBG_ED )
		{
			dbg_p_d_data_i_mn_v2 ( "edge_v2 orig", d_currp, overall_size, wcube_info[0].x,
				wcube_info, nblk_in_x, nblk_in_y ) ;
			dbg_p_d_data_i_mn_v2 ( "edge_v2 get", d_nextp, overall_size, wcube_info[0].x,
				wcube_info, nblk_in_x, nblk_in_y ) ;
		}

		d_currp = d_nextp ;
		d_nextp = the_other_d_buf ( d_currp ) ;

		if (!(i = h_do_copy_box_v2 ( d_currp, d_nextp, overall_size,
			csc.edge_x, csc.edge_y, nblk_in_x, nblk_in_y, d_cs_xyzp, cube_info )))
		{
			printf("%s: copy_box failed i %d size %d \n", __func__, i, overall_size ) ;
			return ( 0 ) ;
		} 

		d_currp = d_nextp ;
		d_nextp = the_other_d_buf ( d_currp ) ;

		for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
        {
			wcube_info[i].x -= csc.edge_x * 2 ;
			wcube_info[i].y -= csc.edge_y * 2 ;

			wcube_info[i].size = wcube_info[i].x *
				wcube_info[i].y * wcube_info[i].z ;
        }

		h_set_config ( d_cs_xyzp, wcube_info ) ;

		inner_cube_size = wcube_info[0].size ;
		overall_size = inner_cube_size * nblk_in_x * nblk_in_y ;

		if ( csc.dbg_flag & DBG_ED )
		{
			dbg_p_d_data_i_mn_v2 ( "edge_v2 after copy", d_currp, overall_size, wcube_info[0].x,
				wcube_info, nblk_in_x, nblk_in_y ) ;
		}

		// motion detection 

#ifdef CUDA_DBG 
		set_device_mem_i( d_nextp, blk_size_e, 111 ) ;
#endif 
		
		k = h_do_motion_idx_v2 ( d_nextp, blk_size_e, &orig,
			nblk_in_x, nblk_in_y, wcube_info,
			csc.md_x, csc.md_y, csc.md_z,
			&rec_size ) ;

#ifdef CUDA_DBG 
		printf("%s: rec_size %d orig %d nblk_in_x/y %d %d \n",
			__func__, rec_size, orig, nblk_in_x, nblk_in_y ) ;
#endif 

		if ( !k )
		{
			printf("%s: motion failed", __func__ ) ;
			return( 0 ) ;
		}

		if ( csc.dbg_flag & DBG_MT_IDX )
			printf("orig idx is %d size %d\n", orig, rec_size ) ;

		hvt_size = ( csc.md_x * 2 + 1 ) * ( csc.md_y * 2 + 1 ) * ( csc.md_z - 1 ) + 1 ;
		
		if ( csc.dbg_flag & DBG_MT_IDX )
			dbg_p_d_data_i_mn ( "idx original ", d_nextp,
				( rec_size + NUM_OF_HVT_INDEX ) * ( hvt_size * nblk_in_x * nblk_in_y ), 
				rec_size + NUM_OF_HVT_INDEX, hvt_size * nblk_in_x * nblk_in_y, 6 ) ;

		omp_timer_on ( CS_TIMER_ANALYSIS_MD0 ) ;

		// step 0 : copy data ...
		k = h_do_motion_detection_step0_v2 ( d_currp, d_nextp,
			rec_size * hvt_size * nblk_in_x * nblk_in_y, 
			rec_size,
			csc.md_x * 2, csc.md_y * 2, csc.md_z,
			d_cs_xyzp,
			hvt_size, inner_cube_size ) ;

		omp_timer_off ( CS_TIMER_ANALYSIS_MD0 ) ;

		if ( !k )
		{
			printf("%s: step0 failed", __func__ ) ;
			return( 0 ) ;
		}

		d_currp = d_nextp ;
		
		if ( csc.dbg_flag & DBG_MT_STEP0 )
			dbg_p_d_data_i_mn ( "motion 0", d_currp,
				( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * nblk_in_x * nblk_in_y, 
				rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

		// step 1 ... yk-y0

		for ( k = 0 ; k < CUBE_INFO_CNT ; k++ )
		{
			wcube_info[k].x -= csc.md_x * 2 ;
			wcube_info[k].y -= csc.md_y * 2 ;
			wcube_info[k].z -= ( csc.md_z - 1 ) ;

			wcube_info[k].size = wcube_info[k].x * wcube_info[k].y * wcube_info[k].z ;
		}

		h_set_config ( d_cs_xyzp, wcube_info ) ;

		// d_nextp = the_other_d_buf ( d_currp ) ;

		omp_timer_on ( CS_TIMER_ANALYSIS_MD1 ) ;

		h_do_l1_norm_step1_v2( d_currp, rec_size * hvt_size * nblk_in_x * nblk_in_y, 
			rec_size, orig, hvt_size ) ;

		omp_timer_off ( CS_TIMER_ANALYSIS_MD1 ) ;

		if ( csc.dbg_flag & DBG_MT_STEP1 )
			dbg_p_d_data_i_mn ( "motion 1", d_currp,
				( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * nblk_in_x * nblk_in_y, 
				rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

		// step 2 -- do the sum

		omp_timer_on ( CS_TIMER_ANALYSIS_MD2 ) ;

		k = h_do_l1_norm_step2_v2( d_currp, rec_size * hvt_size * nblk_in_x * nblk_in_y, 
			rec_size, wcube_info, d_cs_xyzp, d_host_io ) ;

		omp_timer_off ( CS_TIMER_ANALYSIS_MD2 ) ;

		if ( !k )
		{
			printf("%s: step2 failed", __func__ ) ;
			return( 0 ) ;
		}

		if ( csc.dbg_flag & DBG_MT_STEP2 )
			dbg_p_d_data_i_mn ( "motion 2", d_currp,
				( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * nblk_in_x * nblk_in_y, 
				rec_size + NUM_OF_HVT_INDEX, hvt_size, rec_size + NUM_OF_HVT_INDEX ) ;

		// step 3 -- get 1-|y0-yk|/|y0|
		
		omp_timer_on ( CS_TIMER_ANALYSIS_MD3 ) ;

		k = h_do_l1_norm_step3_v2( d_currp, rec_size * hvt_size * nblk_in_x * nblk_in_y,
			rec_size, orig, hvt_size ) ;

		omp_timer_off ( CS_TIMER_ANALYSIS_MD3 ) ;

		printf("%s: step3 done, k %d outbufp %p\n", __func__, k, outbufp ) ;
		if ( !k )
		{
			printf("%s: step3 failed", __func__ ) ;
			return( 0 ) ;
		}

		if ( csc.dbg_flag & DBG_MT_STEP3 )
			dbg_p_d_data_i_mn ( "motion 3", d_currp,
				( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * nblk_in_x * nblk_in_y, 
				rec_size + NUM_OF_HVT_INDEX, hvt_size, 6 ) ;

		// step 4 -- find the winner in each of the blocks

		omp_timer_on ( CS_TIMER_ANALYSIS_MD4 ) ;

		k = h_do_l1_norm_step4_v2( d_currp, rec_size * hvt_size * nblk_in_x * nblk_in_y, 
			rec_size, orig, hvt_size, outbufp, ( csc.md_y * 2 + 1 + 1 ) * csc.md_x ) ;

			// last param is to inidcate the block which is right after the orig
			// in time domain and without the vertical/horizontal shifting.

		omp_timer_off ( CS_TIMER_ANALYSIS_MD4 ) ;

		printf("%s: step4 done k %d\n", __func__, k ) ;
		if ( !k )
		{
			printf("%s: step4 failed", __func__ ) ;
			return( 0 ) ;
		}

		if ( csc.dbg_flag & DBG_MT_STEP4 )
			dbg_p_d_data_i_mn ( "motion 4", d_currp,
				( rec_size + NUM_OF_HVT_INDEX ) * hvt_size * nblk_in_x * nblk_in_y, 
				rec_size + NUM_OF_HVT_INDEX, hvt_size, 8 ) ;

		printf("%s: step4 done\n", __func__ ) ;

		if ( csc.dbg_flag & DBG_MT_STEP4 )
		{
			dbg_p_d_data_i_mn ( "motion 4 host", d_currp,
				( 1 + NUM_OF_HVT_INDEX ) * nblk_in_x * nblk_in_y * 2, 
				( 1 + NUM_OF_HVT_INDEX ) * 2, nblk_in_x * nblk_in_y, ( 1 + NUM_OF_HVT_INDEX ) * 2) ;

			dbg_p_data_i_mn ( "motion 4 host", outbufp,
				( 1 + NUM_OF_HVT_INDEX ) * nblk_in_x * nblk_in_y * 2, 
				( 1 + NUM_OF_HVT_INDEX ) * 2, nblk_in_x * nblk_in_y, ( 1 + NUM_OF_HVT_INDEX ) * 2) ;
		}

#ifdef CUDA_DBG 
		dbg_p_data_i_mn ( "return values", outbufp,
			( 1 + NUM_OF_HVT_INDEX ) * nblk_in_x * nblk_in_y * 2, 
			( 1 + NUM_OF_HVT_INDEX ) * 2, nblk_in_x * nblk_in_y, ( 1 + NUM_OF_HVT_INDEX ) * 2) ;
#endif 
		omp_timer_off ( CS_TIMER_ANALYSIS ) ;
		omp_timer_off( CS_TIMER_TOTAL ) ;
		cs_timer_off( 0 ) ;

		if ( md_filep )
			ma_report_record ( md_filep, outbufp, blocks_processed,
				csc.x_block, csc.y_block, csc.z_block,
        			nblk_in_x, nblk_in_y,
				csc.overlap_x, csc.overlap_y, csc.overlap_z,
				( csc.weight_scheme == WEIGHT_LINEAR )? 1 : 0,
				( csc.weight_scheme == WEIGHT_LINEAR )? 2 : 0 ) ;	// need to #define

		if ( strlen(csc.ipcam_string) )
		{
			cs_ipcam_record ( outbufp, fp->outp ) ;
			cs_ipcam_put ( fp ) ;
		}

	} else
	{
		// NOTE: LDL need work here ... since the size will be different.

		if ( csc.do_swap )
		{
			omp_timer_on( CS_TIMER_SWAP ) ;

			htonl_device_mem_i( d_currp, blk_size_e ) ;

			omp_timer_off( CS_TIMER_SWAP ) ;

			if ( csc.dbg_flag & DBG_SWAP )
				dbg_p_d_data_i ( "after swap", d_currp, blk_size_e ) ;
		}

		outp = outbufp ;
		j = blk_size_i ;

		omp_timer_on( CS_TIMER_MEMCPY_UP ) ;


		if (( i = cudaMemcpy(( char * )outp, d_currp, j,
			cudaMemcpyDeviceToHost)) != cudaSuccess )
		{
			printf("make_one_component:upload fail: %d\n", i ) ;
			return ( 0 ) ;
		}
		omp_timer_off( CS_TIMER_MEMCPY_UP ) ;

		omp_timer_off( CS_TIMER_TOTAL ) ;

		cs_timer_off( 0 ) ;

		if ( csc.do_permutation )
		{
			outp++ ;	// take out 1st entry

			j -= sizeof ( int ) ; 
		}

		// LDL ... prepend block header ... 

		if ( write ( fout, outp, j ) != ( j ))
		{
			printf("make_one_component: write failed errno %d\n",
				errno ) ;
			return ( 0 ) ;
		}
	}

	return ( 1 ) ;
}

int
cs_config_check( struct cs_config *csp )
{
	int err = 0 ;

	if (( csp->adj_x < 0 ) || ( csp->adj_y < 0 ))
	{
		fprintf( stderr, "error: adj %d %d \n", csp->adj_x, csp->adj_y ) ;
		err++ ;
	} else if ( csp->adj_x || csp->adj_y )
		csp->do_block++ ;

	if (( csp->comp_ratio > 100 ) || ( csp->comp_ratio <= 0 ))
	{
		fprintf( stderr, "comp_ratio error %d\n", csp->comp_ratio ) ;
		err++ ;
	}

	if ( csp->do_analysis && (
		( csp->md_x <= 0 ) || 
		( csp->md_y <= 0 ) || 
		( csp->md_z <= 0 )))
	{
		fprintf( stderr, "negative motion detection size %d %d %d \n",
			csp->md_x, csp->md_y, csp->md_z ) ;
		err++ ;
	}

	if ( csp->do_cube && (
		( csp->cubex <= 0 ) || 
		( csp->cubey <= 0 ) || 
		( csp->cubez <= 0 )))
	{
		fprintf( stderr, "negative cube size %d %d %d \n",
			csp->cubex, csp->cubey, csp->cubez ) ;
		err++ ;
	}

	if (( csp->xadd < 0 ) || ( csp->xadd < 0 ) || ( csp->xadd < 0 ))
	{
		fprintf( stderr, "negative expansion size %d %d %d \n",
			csp->xadd, csp->yadd, csp->zadd ) ;
		err++ ;
	}

	if ( csp->do_analysis && (
		( csp->edge_x <= 0 ) || 
		( csp->edge_y <= 0 )))
	{
		fprintf( stderr, "non positive edge size %d %d \n",
			csp->edge_x, csp->edge_y ) ;
		err++ ;
	}

	if ( csp->do_display && (
		( csp->disp_th_x < 0 ) || 
		( csp->disp_th_y < 0 )))
	{
		fprintf( stderr, "negative display threshold %d %d \n",
			csp->disp_th_x, csp->disp_th_y ) ;
		err++ ;
	}

	if (( csp->frame_x <= 0 ) || ( csp->frame_y <= 0 ))
	{
		fprintf( stderr, "frame size error %d %d \n",
			csp->frame_x, csp->frame_y ) ;
		err++ ;
	}

	if ( csp->do_block && (
		( csp->overlap_x < 0 ) || 
		( csp->overlap_y < 0 ) || 
		( csp->overlap_z < 0 )))
	{
		fprintf( stderr, "negative overlap size %d %d %d \n",
			csp->overlap_x, csp->overlap_y, csp->overlap_z ) ;
		err++ ;
	}

	if (( csp->weight_scheme != WEIGHT_LINEAR ) && 
		( csp->weight_scheme != NO_WEIGHT ))
	{
		fprintf( stderr, "weight scheme err %d\n",
			csp->weight_scheme ) ;
		err++ ;
	}

	if ( csp->do_block && (
		( csp->x_block <= 0 ) || 
		( csp->y_block <= 0 ) || 
		( csp->z_block <= 0 )))
	{
		fprintf( stderr, "non positive block size %d %d %d \n",
			csp->x_block, csp->y_block, csp->z_block ) ;
		err++ ;
	}

	//

	if ( csp->do_cube && !csp->do_permutation )
	{   
		fprintf( stderr, "do_cube and not do_permutation\n") ;
		err++ ;
	}   

	if ( csp->ipcam_string && !csp->y_only )
	{   
		fprintf( stderr, "ipcam but not y_only\n") ;
		err++ ;
	}   

	if ( csp->do_cube && (( csp->cubex > csp->x_block ) || ( csp->cubey > csp->y_block )  ||  
		( csp->cubez > csp->z_block )))
	{   
		fprintf( stderr, "Error: cube/block sizes mismatch\n") ;
		err++ ;
	}   

	if ( csp->do_cube && !csp->y_only )
	{
		fprintf( stderr, "do_cube and not y_only\n") ;
		err++ ;
	}

	if ( csp->overlap_z >= csp->z_block )
	{
		fprintf( stderr, "Error: overlap_z %d z_block %d\n",
			csp->overlap_z, csp->z_block ) ;
		err++ ;
	}

	if ( csc.do_permutation && ( !strlen(csc.permdir)))
	{
		fprintf( stderr, "Error: do perm with no perm dir\n") ;
		err++ ;
	}

	if (( csp->frame_x < 0 ) || ( csp->frame_y < 0 ) ||
		( csp->x_block < 0 ) || ( csp->y_block < 0 ) || ( csp->z_block < 0 ) ||
		( !strlen(csp->finname) && !strlen(csp->ipcam_string)) || !strlen(csp->foutname) ||
		( strlen(csp->finname) && strlen(csp->ipcam_string)))
	{

		fprintf( stderr, "Error: misc \n") ;
		err++ ;
	}

	if( strlen(csc.md_outputfile))
	{
		if (( md_filep = fopen ( csc.md_outputfile, "w+")) == NULL )
		{
			fprintf(stderr, "Error: openfile %s\n", csc.md_outputfile ) ;
			err++ ;
		}
	}

	return ( !err ) ;
}
