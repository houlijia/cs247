#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"

#include "cs_matrix.h"
#include "cs_quantize.h"
#include "cs_motion_detect_v4.h"

#define CUDA_DBG
#define CUDA_DBG1

__global__ void d_do_unquan_adj_index ( int *in, int cnt, int noclip, int index_adj, int max )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i ;

	while ( t_idx < cnt )
	{
		i = in[ t_idx ] ;
		i += index_adj ;

		if (( t_idx >= noclip ) && ( i >= max ))
			i = -1 ;

		in[t_idx ] = i ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_adj_index:
	d_in : the int measurements just read in ... need to shift it back to
		its orignal values.  if more than max_out, set it to -1
	
	tbl_size: size in d_in
	index_adj : the number add to the contents ... nbins/2 - 1 
	max_out: if >= max_out, set to -1

	d_in might not be the first entry of the measurements ... skip the DC

NOTE: 1. if nbin is 131, index_adj is ( nbin/2 - 1 ), max_out is nbin
		before -64 .. 66, 67 ( out of range )
		after 0 .. 130, -1 ( out of range )
	  2. the beginning of the d_in starts after the number of no clips ... caller bewared
*/

void 
h_do_unquan_adj_index ( int *d_in, int tbl_size, int noclip, int index_adj, int max_out )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	static int do_dbg_print = 1 ;

#ifdef CUDA_DBG 
	if ( do_dbg_print )
		dbg_p_d_data_i("h_adj_index : before", d_in, tbl_size ) ;
#endif                                                                           
	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_unquan_adj_index <<< nBlocks, nThreadsPerBlock >>> ( d_in, tbl_size, noclip,
		index_adj, max_out ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	h_ck_bin( d_in, tbl_size, max_out, NULL, 0 ) ; 

	if ( do_dbg_print )
	{
		dbg_p_d_data_i("h_adj_index : after", d_in, tbl_size ) ;
		do_dbg_print = 0 ;
	}
#endif                                                                           
}

// skip is noclip ...

int
h_ck_bin ( int *d_in, int size, int max, int *h_zeroed_rows, int skip )
{
	int *bp, err, i ;
	static int *cp = NULL ;

	i = sizeof ( int ) * size ;

	if ( cp == NULL )
		cp = ( int * )malloc ( i ) ;

	get_d_data_i ( d_in, cp, i ) ;

	if ( h_zeroed_rows )
		memset ( h_zeroed_rows, 0, size * sizeof ( int )) ;

	bp = cp ;
	err = 0 ;
	for ( i = 0 ; i < size ; i++ )
	{
		if ((( *bp < 0 ) || ( *bp >= max )) && ( i >= skip ))
		{
			printf("%s: idx %d val %d \n", __func__, i, *bp ) ;

			if ( h_zeroed_rows )
				h_zeroed_rows[err] = i ;
			err++ ;
		}

		bp++ ;
	}

#ifdef CUDA_DBG 
	printf("%s: out of bound %d found \n", __func__, err ) ;
#endif 

	return ( err ) ;
}

__global__ void d_do_unquan_msrmnts ( float *in, int cnt, float intvl, float offset )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	float f ;

	while ( t_idx < cnt )
	{
		f = in[ t_idx ] ;

		if ( f == -1.0 )
			f = 0;
		else
			f = ( f + 1.0 ) * intvl - offset ;	// the 1.0 is to adjust code from matlab to C++
				// in matlab, the index of the vector starts from 1, in C++
				// it starts from 0

		in[ t_idx ] = f ;

		t_idx += CUDA_MAX_THREADS ;
	}
}
/* 
h_do_unquan_msrmnts: unquantize@UniformQuantizer.m l.250~l.252

	soffset = ampl + 0.5 *intvl - qmsr.mean_msr;
    msrmnts = (double(qmsr.bin_numbers) * intvl) - offset;

	d_in and d_out can be the same 
 
NOTE : if d_in and d_out are the same, then the index of the measurement is gone.
*/

// need TO USE THE -1 ... set the outcome to 0 ... LDL LDL LDL LDL LDL
void
h_do_unquan_msrmnts( float *d_inout, int tbl_size, float ampl, float intvl, float mean )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	static int do_dbg_print = 1 ;
	float offset ;
	
	offset = ampl + ( intvl / 2.0 ) - mean ;

	printf("%s: ampl %f offset %f mean %f intv %f\n", __func__, ampl, offset, mean, intvl ) ;

#ifdef CUDA_DBG 
	if ( do_dbg_print )
		dbg_p_d_data_f("h_do_unquan_msrmnts : before", d_inout, tbl_size ) ;
#endif                                                                           

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_unquan_msrmnts <<< nBlocks, nThreadsPerBlock >>> ( d_inout, tbl_size, intvl, offset ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	if ( do_dbg_print )
	{
		dbg_p_d_data_f("h_do_unquan_msrmnts : after", d_inout, tbl_size ) ;
		do_dbg_print = 0 ;
	}
#endif                                                                           
}

// quantization

__global__ void d_setup_for_quant ( struct cube *d_cubep, int cnt,
	int nblk_in_x, int nblk_in_y, float *d_sdp,
	int *d_max_binp, int *d_num_binp, float *d_amplp,
	float *d_offset, float *d_meanp )
{
	int blk_type, t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ( t_idx < cnt )
	{
		blk_type = get_blk_type_idx( t_idx, nblk_in_x, nblk_in_y ) ;

		d_max_binp[ t_idx ] = round (( 3.0 * d_sdp[ t_idx ] ) / d_cubep[ blk_type ].interval ) ;
		d_num_binp[ t_idx ] = 2 * d_max_binp[ t_idx ] + 1 ;

		d_amplp[ t_idx ] = (( float ) d_max_binp[ t_idx ] + 0.5 ) * d_cubep[ blk_type ].interval ;
		d_offset[ t_idx ] = d_amplp[ t_idx ] - d_meanp[ t_idx ] ;
	}
}

__global__ void d_do_quant ( float *od_in, int *od_out, struct cube *d_cubep, int cnt, int max_cnt_in_blk,
	int nblk_in_x, int nblk_in_y, int blk_size,
	float *d_sdp,
	int *d_max_binp, int *d_num_binp, float *d_amplp,
	float *d_offset, float *d_meanp )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int *d_out, i, blk_idx, blk ;
	float *d_in ;

	// QQQ
	while ( t_idx < cnt )
	{
		blk= t_idx / max_cnt_in_blk ;
		blk_idx = get_blk_type_idx( blk, nblk_in_x, nblk_in_y ) ;

		t_idx -= blk * max_cnt_in_blk ; // idx in this block

		if ( t_idx < d_cubep[ blk_idx ].size )
		{
			// points to the beginning of the block
			d_in = od_in + blk * blk_size ;
			d_out = od_out + blk * max_cnt_in_blk ;

			i = round (( d_in[ t_idx ] - d_meanp[ blk ] ) / d_cubep[ blk_idx ].interval ) ;

			if (( i > d_max_binp[ blk ]) || ( i < -d_max_binp[ blk ]))
				i = d_max_binp[ blk ] + 1 ;

			d_out[ t_idx ] = i ;
		}
		t_idx += CUDA_MAX_THREADS ;
	}
}

// do the DC ... i.e. no clip
// dc_inoutp has the input AND output of the data
__global__ void 
d_do_dc_quant ( float *od_inout, struct cube *d_cubep, int cnt, int max_cnt_in_blk,
	int nblk_in_x, int nblk_in_y, int blk_size, float *d_meanp )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, *d_out, blk_idx, blk ;
	float *d_inout ;

	d_out = ( int * )od_inout ;

	// assume only max_cnt_in_blk number of DC's per block.
	while ( t_idx < cnt )
	{
		blk= t_idx / max_cnt_in_blk ;
		blk_idx = get_blk_type_idx( blk, nblk_in_x, nblk_in_y ) ;

		i = t_idx ;

		t_idx -= blk * max_cnt_in_blk ; // idx in this block
		// points to the beginning of the block
		d_inout = od_inout + blk * blk_size ;

		d_out[ i ] = ( int )round (( d_inout[ t_idx ] - d_meanp[ blk ] ) /
			d_cubep[ blk_idx ].interval ) ;

#ifdef CUDA_OBS 
		// d_out[ i ] = ( int )d_inout[ t_idx ] ;
		// d_out[ i ] = ( int ) round( d_meanp[ blk ]) ;
		// d_out[ i ] = ( int )round ( d_inout[ t_idx ] - d_meanp[ blk ] ) ;
		// d_out[ i ] = d_cubep[ blk_idx ].interval ;
#endif 

		t_idx += CUDA_MAX_THREADS ;
	}
}

// setup for the quantization
// mean and sd are known ...
// rest will be updated ...

int
h_do_quant( float *dp, int *d_out, int blk_size,
	struct cube *h_cubep, struct cube *d_cubep,
	int nblk_in_x, int nblk_in_y,
	float *d_meanp, float *d_sdp,
	int *d_max_binp, int *d_num_binp, float *d_amplp, float *d_offset,
	float *d_dcp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	int max_cnt, i ;
	dim3 blk, grid ;

	h_set_cube_config( d_cubep, h_cubep ) ;

	// set up max_bin, num_bin, ample, offset ...

	i = nblk_in_x * nblk_in_y ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_setup_for_quant <<< nBlocks, nThreadsPerBlock >>> ( d_cubep, i, 
		nblk_in_x, nblk_in_y, d_sdp,
		d_max_binp, d_num_binp, d_amplp, d_offset, d_meanp ) ;

	cudaThreadSynchronize() ;

	// do the quant

	max_cnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		if ( max_cnt < h_cubep[i].size ) 
			max_cnt = h_cubep[i].size ; 
	}
	i = max_cnt * nblk_in_x * nblk_in_y ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

#ifdef CUDA_DBG 
	set_device_mem_i ( d_out, i, 7777 ) ;
#endif 

	d_do_quant<<< nBlocks, nThreadsPerBlock >>> ( dp, d_out, d_cubep, i, max_cnt,
		nblk_in_x, nblk_in_y, blk_size,
		d_sdp, d_max_binp, d_num_binp, d_amplp,
		d_offset, d_meanp ) ;

	cudaThreadSynchronize() ;

	i = nblk_in_x * nblk_in_y ;

#ifdef CUDA_DBG 
	// d_dcp will be destroyed ...
	dbg_p_d_data_f("dc:", (float *)d_dcp, i ) ;
#endif 

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_do_dc_quant <<< nBlocks, nThreadsPerBlock >>> ( d_dcp, d_cubep, i, 1,
		nblk_in_x, nblk_in_y, 1, d_meanp ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("dc:interval 0", (float *)&d_cubep[0].interval, 1 ) ;
	dbg_p_d_data_f("dc:interval 1", (float *)&d_cubep[1].interval, 1 ) ;
	dbg_p_d_data_f("dc:interval 2", (float *)&d_cubep[2].interval, 1 ) ;

	dbg_p_d_data_f("dc:mean", ( float *)d_meanp, i ) ;
	dbg_p_d_data_i("dc:quant", ( int *)d_dcp, i ) ;
#endif 

	return ( 1 ) ;
}
