#include <stdio.h>

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_analysis.h"
#include "cs_mean_sd.h"
#include "cs_copy_box.h"
#include "cs_quantize.h"

#define BUF_BLK	100
#define NUM_BLK	9
#define NBLK_IN_X	3
#define NBLK_IN_Y	3

#define CORNER_SIZE	30
#define SIDE_SIZE	60
#define INNER_SIZE	90

#define CUDA_DBG 

#define BUF_SIZE	( BUF_BLK * NUM_BLK )
#define INTER_FACTOR	(1.0)

struct cube h_cube[ CUBE_INFO_CNT ], *d_cubep ;

float *d_meanp, *d_sdp, *d_offset, *d_amplp ;
int *d_max_binp, *d_num_binp, *d_outp ;

float *d_buf, *d_tmp ;
float h_buf[ BUF_SIZE ] ;

main()
{
	int j, i, k ;
	float *ftp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init( 102400 ) ;

	if (( k = cudaMalloc( &d_tmp, sizeof ( int ) * BUF_SIZE * 5 )) != cudaSuccess )
	{
		printf("%s: alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	d_buf = d_tmp + BUF_SIZE ;
	d_cubep = ( struct cube * ) ( d_buf + BUF_SIZE ) ;

	d_meanp = ( float * ) ( d_cubep + CUBE_INFO_CNT ) ;
	d_sdp = d_meanp + NUM_BLK ;
	d_offset = d_sdp + NUM_BLK ;
	d_amplp = d_offset + NUM_BLK ;

	d_max_binp = ( int * ) ( d_amplp + NUM_BLK ) ;
	d_num_binp = d_max_binp + NUM_BLK ;
	d_outp = d_num_binp + NUM_BLK ;

	printf("dtmp %p buf %p mean %p sd %p offset %p amplp %p maxbin %p "
		"num_bin %p output %p\n",
		d_tmp,
		d_buf,
		d_meanp,
		d_sdp,
		d_offset,
		d_amplp,
		d_max_binp,
		d_num_binp,
		d_outp ) ;

	ftp = h_buf ;
	for ( i = 0 ; i < NBLK_IN_X ; i++ )
	{
		for ( k = 0 ; k < NBLK_IN_Y ; k++ )
		{
			for ( j = 0 ; j < BUF_BLK ; j++ )
				*ftp++ = 0.1 + i * 10000 + k * 1000 + j ;
		}
	}

	put_d_data_f ( d_buf, h_buf, sizeof( float ) * BUF_SIZE ) ;

	h_cube[ CUBE_INFO_INNER ].size = INNER_SIZE ;
	h_cube[ CUBE_INFO_SIDE ].size = SIDE_SIZE ;
	h_cube[ CUBE_INFO_CORNER ].size = CORNER_SIZE ;

	h_cube[ CUBE_INFO_INNER].interval = INTER_FACTOR *
		( float )sqrt(( double )h_cube[ CUBE_INFO_INNER].size / ( double )12 ) ;
	h_cube[ CUBE_INFO_SIDE].interval = INTER_FACTOR *
		( float )sqrt(( double )h_cube[ CUBE_INFO_SIDE].size / ( double )12 ) ;
	h_cube[ CUBE_INFO_CORNER].interval = INTER_FACTOR *
		( float )sqrt(( double )h_cube[ CUBE_INFO_CORNER].size / ( double )12 ) ;

	printf("interval %f %f %f \n",
		h_cube[ CUBE_INFO_INNER].interval,
		h_cube[ CUBE_INFO_SIDE].interval,
		h_cube[ CUBE_INFO_CORNER].interval ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("before BLK 0", d_buf, BUF_BLK ) ;
	dbg_p_d_data_f("before BLK 1", d_buf + BUF_BLK, BUF_BLK ) ;
	dbg_p_d_data_f("before BLK 4", d_buf + 4 * BUF_BLK, BUF_BLK ) ;
	dbg_p_d_data_f("before BLK 8", d_buf + ( NUM_BLK-1) * BUF_BLK, BUF_BLK ) ;
#endif 

	if ( !h_mean_sd ( d_buf, d_tmp, h_cube, d_cubep, NBLK_IN_X, NBLK_IN_Y, BUF_BLK, d_meanp, d_sdp ))
	{
		printf("!h_mean_sd failed \n") ;
		exit(3) ;
	}
	
	dbg_p_d_data_f("MEAN", d_meanp, NUM_BLK ) ;
	dbg_p_d_data_f("SD", d_sdp, NUM_BLK ) ;
	
#ifdef CUDA_OBS 
	dbg_p_d_data_f("after BLK 0", d_buf, BUF_BLK ) ;
	dbg_p_d_data_f("after BLK 1", d_buf + BUF_BLK, BUF_BLK ) ;
	dbg_p_d_data_f("after BLK 4", d_buf + 4 * BUF_BLK, BUF_BLK ) ;
	dbg_p_d_data_f("after BLK 8", d_buf + ( NUM_BLK-1) * BUF_BLK, BUF_BLK ) ;
#endif 

	// do quant

	if ( !h_do_quant ( d_buf, d_outp, BUF_BLK, h_cube, d_cubep, NBLK_IN_X, NBLK_IN_Y,
		d_meanp, d_sdp, d_max_binp, d_num_binp, d_amplp, d_offset ))
	{
		printf("h_do_quant failed \n") ;
		exit(4) ;
	}

	dbg_p_d_data_f("AMPL", d_amplp, NUM_BLK ) ;
	dbg_p_d_data_f("OFFSET", d_offset, NUM_BLK ) ;

	dbg_p_d_data_i("MAX", d_max_binp, NUM_BLK ) ;
	dbg_p_d_data_i("NUMBIN", d_num_binp, NUM_BLK ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("after BLK 0", d_outp, INNER_SIZE ) ;
	dbg_p_d_data_i("after BLK 1", d_outp + INNER_SIZE, INNER_SIZE ) ;
	dbg_p_d_data_i("after BLK 4", d_outp + 4 * INNER_SIZE, INNER_SIZE ) ;
	dbg_p_d_data_i("after BLK 8", d_outp + ( NUM_BLK-1) * INNER_SIZE, INNER_SIZE ) ;
#endif 

}
