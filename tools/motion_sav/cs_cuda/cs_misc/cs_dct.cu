#include <stdio.h>
#include <stdlib.h>
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_dbg.h"
#include "cs_dct.h"

#define  CUDA_DBG

// using the DCT to perform the sparsifying transformation

static int init_done = 0 ;

#define MIN_DCT_TBL_SIZE	2
#define MAX_DCT_TBL_SIZE	5

float dct2[]= { 
	0.7071, 0.7071, 
	0.7071, -0.7071 } ;

float dct3[]= {
	0.5774, 0.5774, 0.5774,
	0.7071, 0.0000, -0.7071,
	0.4082, -0.8165, 0.4082 } ;

float dct4[]= {
	0.5000,    0.5000,    0.5000,    0.5000,
	0.6533,    0.2706,   -0.2706,   -0.6533,
	0.5000,   -0.5000,   -0.5000,    0.5000,
	0.2706,   -0.6533,    0.6533,   -0.2706 } ;

float dct5[]= {
	0.4472,    0.4472,    0.4472,    0.4472,    0.4472,
	0.6015,    0.3717,    0.0000,   -0.3717,   -0.6015,
	0.5117,   -0.1954,   -0.6325,   -0.1954,    0.5117,
	0.3717,   -0.6015,   -0.0000,    0.6015,   -0.3717,
	0.1954,   -0.5117,    0.6325,   -0.5117,    0.1954 } ;

static float *dct_result ; // double as idct_result
static float *dct2_dp ;
static float *dct3_dp ;
static float *dct4_dp ;
static float *dct5_dp ;

float *dcttab[4] ;

float idct2[]= { 
	0.7071, 0.7071, 
	0.7071, -0.7071 } ;

float idct3[]= {
	0.5774,    0.7071,    0.4082,
	0.5774,         0,   -0.8165,
	0.5774,   -0.7071,    0.4082 } ;

float idct4[]= {
	0.5000,    0.6533,    0.5000,    0.2706,
	0.5000,    0.2706,   -0.5000,   -0.6533,
	0.5000,   -0.2706,   -0.5000,    0.6533,
	0.5000,   -0.6533,    0.5000,   -0.2706 } ;

float idct5[]= {
	0.4472,    0.6015,    0.5117,    0.3717,    0.1954,
	0.4472,    0.3717,   -0.1954,   -0.6015,   -0.5117,
	0.4472,         0,   -0.6325,    0.0000,    0.6325,
	0.4472,   -0.3717,   -0.1954,    0.6015,   -0.5117,
	0.4472,   -0.6015,    0.5117,   -0.3717,    0.1954 } ;

// static float *idct_result ;
static float *idct2_dp ;
static float *idct3_dp ;
static float *idct4_dp ;
static float *idct5_dp ;

float *idcttab[4] ;

/*
d_do_dcta :::

fdp : from data pointer ... in d mem
tdp : to data pointer ... in d mem
tbl_size : fdp/tdp data size ( in number of int )
dctdp : dct table from one of the above
cx, cy, cz : the data format ... of *fdp, *tdp
	cz is also the size, one dimension, of the dct table
*/

__global__ void d_do_dct ( int *fdp, int *tdp, int tbl_size, float *dctdp,
	int cx, int cy, int cz, float *dct_d_result )
{
	int i, j, k, *fp, *tp ;
	float *dp, d ;
	int xy = cx * cy ;

	int ot_idx, t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// the size is the total size on device

	while ( t_idx < tbl_size )
	{
		ot_idx = t_idx ;

		/// old starts from here ... ldl ... 06/05/14

		i = t_idx / xy ;
		j = t_idx % xy ;

		dp = dctdp + i * cz ;
	   	// fp = fdp + i * xy + j ;
	   	fp = fdp + j ;
	   	tp = tdp + i * xy + j ;

		d = 0.0 ;
		for ( k = 0 ; k < cz ; k++ )
		{
			d += ( *dp ) * ((float)*fp) ;
			dp++ ;
			fp += xy ;
		}
			
		*tp = roundf(d) ;

		t_idx = ot_idx + CUDA_MAX_THREADS ;
	} 
#ifdef CUDA_OBS 
	else
		cudadbgp[ 64 ] = t_idx ;
#endif 

	*dct_d_result = 1.0 ; // good
}

__global__ void d_do_dct ( float *fdp, float *tdp, int tbl_size, float *dctdp,
	int cx, int cy, int cz, float *dct_d_result )
{
	int i, j, k ;
	float *fp, *tp, *dp, d ;
	int xy = cx * cy ;

	int ot_idx, t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// the size is the total size on device

	while ( t_idx < tbl_size )
	{
		ot_idx = t_idx ;

		/// old starts from here ... ldl ... 06/05/14

		i = t_idx / xy ;
		j = t_idx % xy ;

		dp = dctdp + i * cz ;
	   	// fp = fdp + i * xy + j ;
	   	fp = fdp + j ;
	   	tp = tdp + i * xy + j ;

		d = 0.0 ;
		for ( k = 0 ; k < cz ; k++ )
		{
			d += ( *dp ) * ((float)*fp) ;
			dp++ ;
			fp += xy ;
		}
			
		*tp = d ;

		t_idx = ot_idx + CUDA_MAX_THREADS ;
	} 
#ifdef CUDA_OBS 
	else
		cudadbgp[ 64 ] = t_idx ;
#endif 

	*dct_d_result = 1.0 ; // good
}

/* 
input : device addr
output : device addr
xdim : x dimension of frame 
ydim : y dimension of frame
zdim : z dimension of frame // time 

return: 0: fail 1: good
*/
int
h_do_dct ( int *d_input, int *d_output, int xdim, int ydim, int zdim, int inverse )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	float *dct_tbl_p ;
	int nn = xdim * ydim * zdim ;
	int nBlocks ; // = ( nn + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	if ( !init_done )
	{
		fprintf( stderr, "%s: init not done\n", __func__ ) ;
		return ( 0 ) ;
	}


	if (( zdim > MAX_DCT_TBL_SIZE ) || ( zdim < MIN_DCT_TBL_SIZE )) 
		return ( 0 ) ;

	if ( inverse )
		dct_tbl_p = idcttab[ zdim - MIN_DCT_TBL_SIZE ] ;
	else
		dct_tbl_p = dcttab[ zdim - MIN_DCT_TBL_SIZE ] ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: din %p dout %p x/y/z %d %d %d dct_p %p nn %d\n",
		__func__,
		d_input, d_output, xdim, ydim, zdim, dct_tbl_p, nn ) ;
#endif 

	if (( zdim > MAX_DCT_TBL_SIZE ) || ( zdim < MIN_DCT_TBL_SIZE )) 
		return ( 0 ) ;

	// dbg_put_d_data ( d_dct_tbl_p, dct_tbl_p[ zdim - MIN_DCT_TBL_SIZE ] );

	h_block_adj ( nn, nThreadsPerBlock, &nBlocks ) ;

	d_do_dct <<< nBlocks, nThreadsPerBlock >>> ( d_input, d_output,
		xdim * ydim * zdim, dct_tbl_p, xdim, ydim, zdim, dct_result ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

int
h_do_dct ( float *d_input, float *d_output, int xdim, int ydim, int zdim, int inverse )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	float *dct_tbl_p ;
	int nn = xdim * ydim * zdim ;
	int nBlocks ; // = ( nn + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	if ( !init_done )
	{
		fprintf( stderr, "%s: init not done\n", __func__ ) ;
		return ( 0 ) ;
	}

	if (( zdim > MAX_DCT_TBL_SIZE ) || ( zdim < MIN_DCT_TBL_SIZE )) 
		return ( 0 ) ;

	if ( inverse )
		dct_tbl_p = idcttab[ zdim - MIN_DCT_TBL_SIZE ] ;
	else
		dct_tbl_p = dcttab[ zdim - MIN_DCT_TBL_SIZE ] ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: din %p dout %p x/y/z %d %d %d dct_p %p nn %d\n",
		__func__,
		d_input, d_output, xdim, ydim, zdim, dct_tbl_p, nn ) ;
#endif 

	if (( zdim > MAX_DCT_TBL_SIZE ) || ( zdim < MIN_DCT_TBL_SIZE )) 
		return ( 0 ) ;

	// dbg_put_d_data ( d_dct_tbl_p, dct_tbl_p[ zdim - MIN_DCT_TBL_SIZE ] );

	h_block_adj ( nn, nThreadsPerBlock, &nBlocks ) ;

	d_do_dct <<< nBlocks, nThreadsPerBlock >>> ( d_input, d_output,
		xdim * ydim * zdim, dct_tbl_p, xdim, ydim, zdim, dct_result ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

int
h_do_dct_init ()
{
	float *dp ;
	int j, i ;

	i = sizeof ( dct2 ) +
		sizeof ( dct3 ) +
		sizeof ( dct4 ) +
		sizeof ( dct5 )  ;

	if (( j = cudaMalloc( &dp, i * 2 + 100 )) != cudaSuccess )
	{
		fprintf( stderr, "%s: cudaMalloc failed %d\n", __func__, j ) ;
		return ( 0 ) ;
	}

	printf("%s: dct table %lx, size %d -- %lu \n", __func__,
	       (unsigned long)dp, i, (unsigned long)(i / sizeof ( float ))) ;

	dct_result = dp++ ;

	dct2_dp = dp ;
	cudaMemcpy( dp, dct2, sizeof ( dct2 ), cudaMemcpyHostToDevice ) ;
	dp += sizeof ( dct2 ) / sizeof ( float ) ;

	dct3_dp = dp ;
	cudaMemcpy( dp, dct3, sizeof ( dct3 ), cudaMemcpyHostToDevice ) ;
	dp += sizeof ( dct3 ) / sizeof ( float ) ;

	dct4_dp = dp ;
	cudaMemcpy( dp, dct4, sizeof ( dct4 ), cudaMemcpyHostToDevice ) ;
	dp += sizeof ( dct4 ) / sizeof ( float ) ;

	dct5_dp = dp ;
	cudaMemcpy( dp, dct5, sizeof ( dct5 ), cudaMemcpyHostToDevice ) ;

	dcttab[0] = dct2_dp ;
	dcttab[1] = dct3_dp ;
	dcttab[2] = dct4_dp ;
	dcttab[3] = dct5_dp ;

#ifdef CUDA_OBS 
	dbg_p_d_data_d ("h_do_dct_init:2", dcttab[0], sizeof ( dct2 )) ;
	dbg_p_d_data_d ("h_do_dct_init:3", dcttab[1], sizeof ( dct3 )) ;
	dbg_p_d_data_d ("h_do_dct_init:4", dcttab[2], sizeof ( dct4 )) ;
	dbg_p_d_data_d ("h_do_dct_init:5", dcttab[3], sizeof ( dct5 )) ;
#endif 

	// idct tables 

	dp += sizeof ( dct5 ) / sizeof ( float ) ;

	// idct_result = dp++ ;

	idct2_dp = dp ;
	cudaMemcpy( dp, idct2, sizeof ( idct2 ), cudaMemcpyHostToDevice ) ;
	dp += sizeof ( idct2 ) / sizeof ( float ) ;

	idct3_dp = dp ;
	cudaMemcpy( dp, idct3, sizeof ( idct3 ), cudaMemcpyHostToDevice ) ;
	dp += sizeof ( idct3 ) / sizeof ( float ) ;

	idct4_dp = dp ;
	cudaMemcpy( dp, idct4, sizeof ( idct4 ), cudaMemcpyHostToDevice ) ;
	dp += sizeof ( idct4 ) / sizeof ( float ) ;

	idct5_dp = dp ;
	cudaMemcpy( dp, idct5, sizeof ( idct5 ), cudaMemcpyHostToDevice ) ;

	idcttab[0] = idct2_dp ;
	idcttab[1] = idct3_dp ;
	idcttab[2] = idct4_dp ;
	idcttab[3] = idct5_dp ;

#ifdef CUDA_OBS 
	dbg_p_d_data_d ("h_do_dct_init:2", idcttab[0], sizeof ( idct2 )) ;
	dbg_p_d_data_d ("h_do_dct_init:3", idcttab[1], sizeof ( idct3 )) ;
	dbg_p_d_data_d ("h_do_dct_init:4", idcttab[2], sizeof ( idct4 )) ;
	dbg_p_d_data_d ("h_do_dct_init:5", idcttab[3], sizeof ( idct5 )) ;
#endif 

	init_done = 1 ;
	
	return ( 1 ) ;
}	
