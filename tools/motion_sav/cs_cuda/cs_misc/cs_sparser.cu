#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_cuda.h"
#include "cs_dbg.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_helper.h"
#include "cs_decode_misc.h"
#include "cs_sparser.h"
#include "cs_dct.h"
#include "cs_matrix.h"

#define CUDA_DBG
// #define CUDA_DBG1

#define DBG_MARKER	77.77

/* 
h_compDiffTrnspPx1_v:
	// in .m; input size is 71x88x2 x3 ... but in .cu it is 72x88x2 x3
	- set 0 at the bottom to make it 72x88x2

*/
	
__global__ void
d_compDiffTrnspPx1_v( float *d_in, float *d_out, int size, int h, int v, int hv )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int row ;

	while ( t_idx < size )
	{
		row = ( t_idx % hv ) / h ;

		if ( row == 0 )
		{
			*( d_out + t_idx ) = - *( d_in + t_idx ) ;
		} else if ( row == ( v - 1 ))
		{
			*( d_out + t_idx ) = *( d_in + t_idx - h ) ;
		}else 
		{
			*( d_out + t_idx ) = *( d_in + t_idx - h ) -
				*( d_in + t_idx ) ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

/* 
h_compDiffTrnspPx1_v:
	d_in : should have 72x88x2 x3
		althou the last row of every block is not used, the real data 
		is actually 71x88x2 x3
	d_out : all entries are used ...
		if d_in is [ a b c d e X ] 
		d_out is [ -a a-b b-c c-d d-e e ]

	h, v, t : 
	c : colors
*/

int
h_compDiffTrnspPx1_v( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

#ifdef CUDA_DBG 
	printf("%s: in %p out %p size %d h %d v %d t %d c %d \n", __func__, d_in,
		d_out, tsize, vhtcp->h, vhtcp->v, vhtcp->t, vhtcp->c ) ;
#endif 

	if ( vhtcp->size != tsize )
	{
#ifdef CUDA_DBG 
		printf("%s: tsize %d size %d ", __func__, tsize, vhtcp->size ) ;
#endif    
		return ( 0 ) ;
	}

	h_block_adj ( vhtcp->size, nThreadsPerBlock, &nBlocks ) ;

	d_compDiffTrnspPx1_v <<< nBlocks, nThreadsPerBlock >>> ( d_in, d_out, vhtcp->size,
		vhtcp->h, vhtcp->v, vhtcp->h * vhtcp->v ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}


/* 
h_compDiffTrnspPx1_h:
	// in .m; input size is 71x88x2 x3 ... but in .cu it is 72x88x2 x3
	- set 0 at the bottom to make it 72x88x2

*/
	
__global__ void
d_compDiffTrnspPx1_h( float *d_in, float *d_out, int size, int h )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int column ;

	while ( t_idx < size )
	{
		column = t_idx % h ;

		if ( column == 0 )
		{
			*( d_out + t_idx ) = - *( d_in + t_idx ) ;
		} else if ( column == ( h - 1 ))
		{
			*( d_out + t_idx ) = *( d_in + t_idx - 1 ) ;
		}else 
		{
			*( d_out + t_idx ) = *( d_in + t_idx - 1 ) -
				*( d_in + t_idx ) ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

/* 
h_compDiffTrnspPx1_h:
	d_in : should have 72x88x2 x3
		althou the last column of every block is not used, the real data 
		is actually 72x87x2 x3
	d_out : all entries are used ...
		if d_in is [ a b c d e X ] in v 
		d_out is [ -a a-b b-c c-d d-e e ] in v

	h, v, t : 
	c : colors
*/

int
h_compDiffTrnspPx1_h( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

#ifdef CUDA_DBG 
	printf("%s: in %p out %p size %d h %d v %d t %d c %d \n", __func__, d_in,
		d_out, tsize, vhtcp->h, vhtcp->v, vhtcp->t, vhtcp->c ) ;
#endif 

	if ( vhtcp->size != tsize )
	{
#ifdef CUDA_DBG 
		printf("%s: tsize %d size %d ", __func__, tsize, vhtcp->size ) ;
#endif    
		return ( 0 ) ;
	}

	h_block_adj ( vhtcp->size, nThreadsPerBlock, &nBlocks ) ;

	d_compDiffTrnspPx1_h <<< nBlocks, nThreadsPerBlock >>> ( d_in, d_out, vhtcp->size, vhtcp->h ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

// VidSparserTV.m  ... do_compSprsVec()

/* 
d_compDiffInside_v:
	d_in : 72x88x2 x3
	d_out: in .m, it is 71x88x2 x3, in .cu it is 72x88x2 x3 ... the last row is junk 
*/
	
__global__ void
d_compDiffInside_v( float *d_in, float *d_out, int size, int h, int v, int hv )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int row ;

	while ( t_idx < size )
	{
		row = ( t_idx % hv ) / h ;

		if ( row < ( v - 1 ))
		{
			*( d_out + t_idx ) = - *( d_in + t_idx ) +
				*( d_in + t_idx + h ) ;
		} else
			*( d_out + t_idx ) = DBG_MARKER ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/* 
h_compDiffInside_v:
	d_in : should have 72x88x2 x3
	d_out : last row of each block is junk
	h, v, t : 
	c : colors
*/

int
h_compDiffInside_v( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

	if ( vhtcp->size != tsize )
	{
#ifdef CUDA_DBG 
		printf("%s: tsize %d size %d ", __func__, tsize, vhtcp->size ) ;
#endif    
		return ( 0 ) ;
	}

	h_block_adj ( vhtcp->size, nThreadsPerBlock, &nBlocks ) ;

	d_compDiffInside_v <<< nBlocks, nThreadsPerBlock >>> ( d_in, d_out, vhtcp->size, vhtcp->h,
		vhtcp->v, vhtcp->h * vhtcp->v ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

/* 
h_compDiffInside_h:
	d_in : 72x88x2 x3
	d_out : in .m, outputsize is 71x88x2 x3 ... but in .cu it is 72x88x2 x3
	the last column is junk

	if d_in has [ a b c d e ]
	then d_out will have [ b-a c-b d-c e-d ] and lose 1 row or 1 column
*/
	
__global__ void
d_compDiffInside_h( float *d_in, float *d_out, int size, int h )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int column ;

	while ( t_idx < size )
	{
		column = t_idx % h ;

		if ( column < ( h - 1 ))
		{
			*( d_out + t_idx ) = *( d_in + t_idx + 1 ) -
				*( d_in + t_idx ) ;
		} else
			*( d_out + t_idx ) = DBG_MARKER ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/* 
h_compDiffInside_h:
	d_in : should have 72x88x2 x3
	d_out : all entries are used ...
		if d_in is [ a b c d e ]  
		d_out is [ b-a c-b d-c e-d X ] 

	h, v, t : 
	c : colors
*/

int
h_compDiffInside_h( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

	if ( vhtcp->size != tsize )
	{
#ifdef CUDA_DBG 
		printf("%s: tsize %d size %d ", __func__, tsize, vhtcp->size ) ;
#endif    
		return ( 0 ) ;
	}

	h_block_adj ( vhtcp->size, nThreadsPerBlock, &nBlocks ) ;

	d_compDiffInside_h <<< nBlocks, nThreadsPerBlock >>> ( d_in, d_out, vhtcp->size, vhtcp->h ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

/*
h_compSprsVec:
	calculate the sparce vector Dx-w
	d_input : 72x88x2 x3 ... 38016	// not changed
	d_tmp : tmp buf	72x88x2 x3 ... 38016
	dvh_output : 72x88x2 x3 x2
		// vertical ... lost one row	
		// horizontal ... lost one column

	cf. VidSparserTV_DCT.do_compSprsVec()

*/

int
h_compSprsVec( float *d_input, float *d_tmp, float *dvh_output,
	struct vhtc *vhtcp, int tsize )
{
	int i ;
	int blk_size = vhtcp->h * vhtcp->v * vhtcp->t ;

	if ( vhtcp->size != tsize )
	{
#ifdef CUDA_DBG 
		printf("%s: tsize %d size %d ", __func__, tsize, vhtcp->size ) ;
#endif    
		return ( 0 ) ;
	}

	for ( i = 0 ; i < vhtcp->c ; i++ )
		h_do_dct( d_input + i * blk_size, d_tmp + i * blk_size, vhtcp->h, vhtcp->v, vhtcp->t, 0 ) ;

	h_compDiffInside_v( d_tmp, dvh_output, vhtcp->size,
		vhtcp ) ;	// 71x88x2 x3
	h_compDiffInside_h( d_tmp, dvh_output + vhtcp->size, vhtcp->size,
		vhtcp ) ;	// 72x87x2 x3

	h_do_cleanup_total_value_vh_elements( dvh_output, vhtcp ) ;

	return ( 1 ) ;
}

/*
h_compSprsVecTrnsp:  cf. compSprsVecTrnsp@BaseSparser.m

	d_in : // in .m, 75072 but in .cu 72x88x2 x3 x2 format ...
	size : v * h * t * c * 2 // should equal to this
	d_out : the first v * h * t * c entries are valid ... 72x88x2 x3	// cjerr.D
	d_tmp : buf size of v * h * t * c
*/

int
h_compSprsVecTrnsp( float *d_in, float *d_out, float *d_tmp, int size, struct vhtc *vhtcp )
{
	int i, tsize ;
	int blk_size = vhtcp->h * vhtcp->v * vhtcp->t ;

	tsize = vhtcp->h * vhtcp->v * vhtcp->t * vhtcp->c * 2 ;	// one for v ... one for h ...

	if ( tsize != ( vhtcp->size * 2 ))
	{
		printf("%s: tsize %d v %d h %d t %d c %d \n", __func__, tsize,
			vhtcp->v, vhtcp->h, vhtcp->t, vhtcp->c ) ;
		return ( 0 ) ;
	}

	tsize >>= 1 ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp before element v", d_in, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp before element h", d_in + tsize, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	/* NO NEED TO CLEAN UP first

	h_do_cleanup_total_value_vh_elements ( d_in, vhtcp, 1 ) ;
	h_do_cleanup_total_value_vh_elements ( d_in + tsize, vhtcp, 0 ) ;

	NO NEED TO CLEAN UP first */

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp after element v", d_in, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp after element h", d_in + tsize, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	if ( !h_compDiffTrnspPx1_v ( d_in, d_tmp, tsize, vhtcp ))
	{
		printf("%s: fail v tsize %d v %d h %d t %d c %d \n", __func__, tsize,
			vhtcp->v, vhtcp->h, vhtcp->t, vhtcp->c ) ;
		return ( 0 ) ;
	}

	if ( !h_compDiffTrnspPx1_h ( d_in + tsize, d_out, tsize, vhtcp ))
	{
		printf("%s: fail h tsize %d v %d h %d t %d c %d \n", __func__, tsize,
			vhtcp->v, vhtcp->h, vhtcp->t, vhtcp->c ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp v", d_tmp, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp h", d_out, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	h_do_vector_add_vector( d_tmp, d_out, d_tmp, tsize ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp after add", d_tmp, tsize,
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	for ( i = 0 ; i < vhtcp->c ; i++ )
		h_do_dct( d_tmp + i * blk_size, d_out + i * blk_size, vhtcp->h, vhtcp->v, vhtcp->t, 1 ) ;	// inverse

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "h_compSprsVecTrnsp after idct", d_out, tsize, vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	return ( 1 ) ;	

	// need no to vectorize as in .m
}

__global__ void
d_do_max_sign_in_optimize( float *d_in, float beta_inv, int size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int sign ;
	float d ;

	while ( t_idx < size )
	{
		d = d_in[ t_idx ] ;

		if ( d < 0.0 )
		{
			sign = -1.0 ;
			d = -d  ;
		} else
			sign = 1.0 ;

		d -= beta_inv ;

		if ( d < 0.0 )
			d = 0.0 ;

		d_in[ t_idx ] = d * sign ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_max_sign_in_optimize:
	dvh_input : 72x88x2 x3 x2	// basically, Dx vector
		// vertical ... lost one row	
		// horizontal ... lost one column

	dvh_size : h * v * t * c * 2 // one for v one for h

	cf. BaseSparser.m:optimize

*/
void
h_do_max_sign_in_optimize( float *dvh_input, float beta_inv, int dvh_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

#ifdef CUDA_DBG 
	printf("%s: d_in %p size %d ", __func__, dvh_input, dvh_size ) ;
#endif    

	h_block_adj ( dvh_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_max_sign_in_optimize <<< nBlocks, nThreadsPerBlock >>> ( dvh_input, beta_inv, dvh_size ) ;

	cudaThreadSynchronize() ;
}

/*
h_optimize:
	dvh_input : 72x88x2 x3 x2	// basically, Dx vector
		// vertical ... lost one row	
		// horizontal ... lost one column
	dvh_output : 72x88x2 x3 x2
		// vertical ... lost one row	
		// horizontal ... lost one column

	dvh_size : h * v * t * c * 2 // one for v one for h
	d_mltplr: h*v*t*c*2 ... this is lambda_D ... 75072 in .m 

dvh_out : the sprs_vec, w, in optimize @ basesparse.m

	cf. BaseSparser.m:optimize

*/
void 
h_optimize( float *dvh_input, float *dvh_out, float *d_mltplr, float beta, int dvh_size)
{
	float beta_inv ;

	beta_inv = 1.0 / beta ; 

	// lambda = mltplr * beta_inv ;

	h_do_scale_mul_vector( d_mltplr, beta_inv, dvh_size, dvh_out ) ;

	// h_do_copy_vector( dvh_input, dvh_out, dvh_size ) ;	

	h_do_vector_add_vector ( dvh_input, dvh_out, dvh_out, dvh_size ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_optimize : after h_do_vector_add_vector ", dvh_out, 10 ) ;
#endif 

	/* only 71x88x2 x3 and 72x87x2 x3 have valid data, althou it is represented as 
	   72x88x2 x3 x2 here ... so there is junk in some of the elements */

	h_do_max_sign_in_optimize( dvh_out, beta_inv, dvh_size ) ;
}

/* 
h_do_Jopt: cf optimize_solver_w.m 

	wvec : 75072 in .m ... but in here is 76032 72x88x2 x3 ...
*/

float 
h_do_Jopt( float *dvh_input, float *dvh_tmp, struct vhtc *vhtcp )
{
	int total_size = vhtcp->size * 2 ;
	float i ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "Jopt before v", dvh_input, vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "Jopt before h", dvh_input + vhtcp->size,
		vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

#ifdef CUDA_OBS 
	// no need ...
	h_do_cleanup_total_value_vh_elements( dvh_input, vhtcp, 1 ) ;
	h_do_cleanup_total_value_vh_elements( dvh_input + vhtcp->size, vhtcp, 0 ) ;
#endif 

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "Jopt after v", dvh_input, vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "Jopt after h", dvh_input + vhtcp->size,
		vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	h_do_copy_vector( dvh_input, dvh_tmp, total_size ) ;	

	h_do_abs_vector ( dvh_tmp, total_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn ( "Jopt abs v", dvh_tmp, vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "Jopt abs h", dvh_tmp +  vhtcp->size,
		vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	i = h_do_vector_add_destroy ( dvh_tmp, total_size ) ;

#ifdef CUDA_DBG 
	printf("%s: Jopt %f \n", __func__, i ) ;
#endif 
	return ( i ) ;
}


// to clean up excess elements in the matrix ... when we do the total value logic

__global__ void
d_do_cleanup_total_value_vh_elements_v( float *d_in, int size, int v, int h )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int column, blk ;

	while ( t_idx < size )
	{
		blk = t_idx / h ;

	    column = t_idx % h ; 	

		d_in[ blk * h * v + h * ( v - 1 ) + column ] = 0.0 ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

__global__ void
d_do_cleanup_total_value_vh_elements_h( float *d_in, int size, int v, int h )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int row, blk ;

	while ( t_idx < size )
	{
		blk = t_idx / v ;

	    row = t_idx % v ; 	

		d_in[ blk * h * v + h * ( row + 1 ) - 1 ] = 0.0 ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_max_sign_in_optimize:
	dvh_input : 72x88x2 x3 x2	// basically, Dx vector
		// vertical ... lost one row	
		// horizontal ... lost one column

	dvh_size : h * v * t * c * 2 // one for v one for h

	cf. BaseSparser.m:optimize

*/
void
h_do_cleanup_total_value_vh_elements( float *dvh_input, struct vhtc *vhtcp, int do_v )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;
	int size ;

	if ( do_v )
		size = vhtcp->h * vhtcp->t * vhtcp->c ;
	else
		size = vhtcp->v * vhtcp->t * vhtcp->c ;

#ifdef CUDA_DBG 
	printf("%s: d_in %p size %d v %d h %d t %d c %d do_v %d \n", __func__, dvh_input, size,
		 vhtcp->v, vhtcp->h, vhtcp->t, vhtcp->c, do_v ) ;
#endif    

	h_block_adj ( size, nThreadsPerBlock, &nBlocks ) ;

	if ( do_v )
		d_do_cleanup_total_value_vh_elements_v <<< nBlocks, nThreadsPerBlock >>> ( dvh_input,
			size, vhtcp->v, vhtcp->h ) ;
	else
		d_do_cleanup_total_value_vh_elements_h <<< nBlocks, nThreadsPerBlock >>> ( dvh_input,
			size, vhtcp->v, vhtcp->h ) ;

	cudaThreadSynchronize() ;
}

void
h_do_cleanup_total_value_vh_elements( float *dvh_input, struct vhtc *vhtcp )
{
	h_do_cleanup_total_value_vh_elements( dvh_input, vhtcp, 1 ) ;
	h_do_cleanup_total_value_vh_elements( dvh_input + vhtcp->size, vhtcp, 0 ) ;
}

/*
h_optimize_solver_w: cf h_optimize_solver_w@optimize_solver_w.m

	d_xvec: 38016 ... v*h*t*c
	beta_D:
	d_lambda_D: 75072 ... in our case it is v*h*t*c x2
	d_wvec_ref: 75072 ... but is v*h*t*c x2

	d_tmp : 75072 ... v*h*t*c x2 ... 

	d_dxerr_D: 75072
	Jopt: 1
	d_wvec: 75072
	d_Dxvec: 75072
*/

int
h_optimize_solver_w( float *d_xvec, float beta_D, float *d_lambda_D, float *d_wvec_ref,
	float *J_opt, float *d_wvec, float *d_dxerr_D, float *d_tmp, float *d_Dxvec,
	struct vhtc *vhtcp )
{
	
	if ( !h_compSprsVec( d_xvec, d_tmp, d_Dxvec, vhtcp, vhtcp->size ))
	{
		printf("%s: h_compSprsVec failed \n", __func__ ) ;
		return ( 0 ) ;	
	}

#ifdef CUDA_DBG 
	// for test/cs_sparser_test.cu only
	dbg_p_d_data_f_mn ( "h_compSprsVec d_xvec", d_xvec, vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec d_tmp", d_tmp, vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec d_Dxvec v", d_Dxvec, vhtcp->size, vhtcp->h, vhtcp->v, vhtcp->h ) ;
	dbg_p_d_data_f_mn ( "h_compSprsVec d_Dxvec h", d_Dxvec + vhtcp->size, vhtcp->size, 
		vhtcp->h, vhtcp->v, vhtcp->h ) ;
#endif 

	if ( d_wvec_ref )
	{
		h_do_copy_vector ( d_wvec_ref, d_wvec, vhtcp->size_x2 ) ;
	} else
		h_optimize ( d_Dxvec, d_wvec, d_lambda_D, beta_D, vhtcp->size_x2 ) ;

	*J_opt = h_do_Jopt( d_wvec, d_tmp, vhtcp ) ;

	h_do_vector_sub_vector( d_Dxvec, d_wvec, d_dxerr_D, vhtcp->size_x2 ) ;

	return ( 1 ) ;
}
