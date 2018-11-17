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

// #define CUDA_DBG
#define CUDA_DBG1

__global__ void d_do_vector_add_destroy ( int *in1, int to_idx, int from_idx )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < to_idx )
	{
		if (( t_idx >= from_idx ) && ( t_idx ))
			in1[ t_idx - from_idx ] += in1[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_vector_add_destroy: will add the value of all entries in the vector
the contents of the vector will be destroyed in the process.
the first entry of the vector will have the sum
*/

int
h_do_vector_add_destroy ( int *d_in, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int j, i, nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	i = max_log2( tbl_size ) ;

	while ( i != 1 )
	{
#ifdef CUDA_OBS 
		dbg_p_d_data_i("h_do_vector_add_destroy before", d_in, tbl_size ) ;
#endif                                                                           
		j = ( i >> 1 ) ;

		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_vector_add_destroy <<< nBlocks, nThreadsPerBlock >>> ( d_in, tbl_size, j ) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
		dbg_p_d_data_i("h_do_vector_add_destroy after", d_in, tbl_size ) ;
#endif                                                                           
		tbl_size = j ;
		i >>= 1 ;
	}
	get_d_data_i ( d_in, &i, sizeof( i )) ; 
	return ( i ) ;
}

__global__ void d_do_vector_add_destroy ( float *in1, int to_idx, int from_idx )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < to_idx )
	{
		if (( t_idx >= from_idx ) && ( t_idx ))
			in1[ t_idx - from_idx ] += in1[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_vector_add_destroy: will add the value of all entries in the vector
the contents of the vector will be destroyed in the process.
the first entry of the vector will have the sum
*/

float
h_do_vector_add_destroy ( float *d_in, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int j, i, nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	float ret ;

	i = max_log2( tbl_size ) ;

	while ( i != 1 )
	{
#ifdef CUDA_OBS 
		dbg_p_d_data_i("h_do_vector_add_destroy before", d_in, tbl_size ) ;
#endif                                                                           
		j = ( i >> 1 ) ;

		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_vector_add_destroy <<< nBlocks, nThreadsPerBlock >>> ( d_in, tbl_size, j ) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
		dbg_p_d_data_i("h_do_vector_add_destroy after", d_in, tbl_size ) ;
#endif                                                                           
		tbl_size = j ;
		i >>= 1 ;
	}
	get_d_data_f ( d_in, &ret, sizeof( ret )) ; 
	return ( ret ) ;
}

__global__ void d_do_dot ( int *in1, int *in2, int *out,
	int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		out[ t_idx ] = in1[ t_idx ] * in2[ t_idx ] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

/* 
   h_do_dot will implement the dot() in matlab.  
   tbl_size can be even/odd, not have to be any power of 2 
   d_tmp will have garbage when done
*/

int
h_do_dot ( int *d_input1, int *d_input2, int *d_tmp, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int i, nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_OBS 
	printf("%s: din1 %p din2 %p out %p tblsize %d\n", __func__,
		d_input1, d_input2, d_tmp, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_dot <<< nBlocks, nThreadsPerBlock >>> ( d_input1, d_input2,
		d_tmp, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("input 1", d_input1, tbl_size ) ;
	dbg_p_d_data_i("input 2", d_input2, tbl_size ) ;
	dbg_p_d_data_i("tmp", d_tmp, tbl_size ) ;
#endif                                                                           

	i = h_do_vector_add_destroy( d_tmp, tbl_size ) ;

	return ( i ) ;
}

__global__ void d_do_dot ( float *in1, float *in2, float *out,
	int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		out[ t_idx ] = in1[ t_idx ] * in2[ t_idx ] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

float
h_do_dot ( float *d_input1, float *d_input2, float *d_tmp, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	float f ;

#ifdef CUDA_OBS 
	printf("%s: din1 %p din2 %p out %p tblsize %d\n", __func__,
		d_input1, d_input2, d_tmp, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_dot <<< nBlocks, nThreadsPerBlock >>> ( d_input1, d_input2,
		d_tmp, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("input 1", d_input1, tbl_size ) ;
	dbg_p_d_data_f("input 2", d_input2, tbl_size ) ;
	dbg_p_d_data_f("tmp", d_tmp, tbl_size ) ;
#endif                                                                           

	f = h_do_vector_add_destroy( d_tmp, tbl_size ) ;

	return ( f ) ;
}

__global__ void d_do_scale_mul_vector ( float *in1, float scale, int tbl_size, float *out )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		out[ t_idx ] = in1[ t_idx ] * scale ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

__global__ void d_do_scale_mul_vector ( int *in1, int scale, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		in1[ t_idx ] *= scale ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
   h_do_scale_mul_vector: will multiply all entries in the vector with scale
*/

void
h_do_scale_mul_vector( int *d_in1, int scale, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p scale %f tblsize %d\n", __func__,
		d_in1, scale, tbl_size ) ;
	dbg_p_d_data_i("h_do_scale_mul_vector before", d_in1, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_scale_mul_vector <<< nBlocks, nThreadsPerBlock >>> ( d_in1, scale, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("h_do_scale_mul_vector after", d_in1, tbl_size ) ;
#endif                                                                           
}

void
h_do_scale_mul_vector( float *d_in1, float scale, int tbl_size, float *d_out )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p scale %f tblsize %d\n", __func__,
		d_in1, scale, tbl_size ) ;
	// dbg_p_d_data_i("h_do_scale_mul_vector before", d_in1, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_scale_mul_vector <<< nBlocks, nThreadsPerBlock >>> ( d_in1, scale, tbl_size, d_out ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	// dbg_p_d_data_i("h_do_scale_mul_vector after", d_out, tbl_size ) ;
#endif                                                                           
}

template<typename T>
__global__ void d_do_scale_add_vector ( T *in1, T scale, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		in1[ t_idx ] += scale ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_scale_add_vector
   :: vector add a constant
 
*/
template<typename T>
void
h_do_scale_add_vector( T *d_in1, T toadd, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p scale %f tblsize %d\n", __func__,
		d_in1, toadd, tbl_size ) ;

	// dbg_p_d_data_f("h_do_scale_add_vector", ( float *)d_in1, tbl_size ) ;
	// dbg_p_d_data_i("h_do_scale_add_vector", ( int *)d_in1, tbl_size ) ;

#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_scale_add_vector<T> <<< nBlocks, nThreadsPerBlock >>> ( d_in1, toadd, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	// dbg_p_d_data_i("h_do_scale_add_vector", ( int *)d_in1, tbl_size ) ;
	// dbg_p_d_data_f("h_do_scale_add_vector", ( float *)d_in1, tbl_size ) ;
#endif                                                                           
}

template void h_do_scale_add_vector<int>( int *d_in1, int toadd, int tbl_size ) ;
template void h_do_scale_add_vector<float>( float *d_in1, float toadd, int tbl_size ) ;

// dup a vector

__global__ void d_do_copy_vector ( float *in1, float *out, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		out[ t_idx ] = in1[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_copy_vector
*/

void
h_do_copy_vector( float *d_in1, float *d_out, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p out %p tblsize %d\n", __func__,
		d_in1, d_out, tbl_size ) ;

	// dbg_p_d_data_f("h_do_copy_vector", d_in1, tbl_size ) ;

#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_copy_vector <<< nBlocks, nThreadsPerBlock >>> ( d_in1, d_out, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_copy_vector OUT", d_out, tbl_size ) ;
#endif                                                                           
}

// abs all the elements 

__global__ void d_do_abs_vector ( float *in1, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		if ( in1[ t_idx ] < 0 )
			in1[ t_idx ] = - in1[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_copy_vector
	: original vector value is gone ...
*/

void
h_do_abs_vector( float *d_in1, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p tblsize %d\n", __func__,
		d_in1, tbl_size ) ;

	// dbg_p_d_data_f("h_do_abs_vector", d_in1, tbl_size ) ;

#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_abs_vector <<< nBlocks, nThreadsPerBlock >>> ( d_in1, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_abs_vector OUT", d_out, tbl_size ) ;
#endif                                                                           
}


// vector3 = vector1 - vector2 ;

__global__ void d_do_vector_sub_vector ( float *d_in1, float *d_in2, float *d_out, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		d_out[ t_idx ] = d_in1[ t_idx ] - d_in2[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_vector_sub_vector
	: d_out = d_in1 - d_in2 ;
*/

void
h_do_vector_sub_vector ( float *d_in1, float *d_in2, float *d_out, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p din2 %p out %p tblsize %d\n", __func__,
		d_in1, d_in2, d_out, tbl_size ) ;

	// dbg_p_d_data_f("h_do_vector_sub_vector 1", d_in1, tbl_size ) ;
	// dbg_p_d_data_f("h_do_vector_sub_vector 2", d_in2, tbl_size ) ;

#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_vector_sub_vector <<< nBlocks, nThreadsPerBlock >>> ( d_in1, d_in2, d_out, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_vector_sub_vector OUT", d_out, tbl_size ) ;
#endif                                                                           
}

// vector3 = vector1 + vector2 ;

__global__ void d_do_vector_add_vector ( float *d_in1, float *d_in2, float *d_out, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		d_out[ t_idx ] = d_in1[ t_idx ] + d_in2[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_vector_add_vector
	: d_out = d_in1 + d_in2 ;
*/

void
h_do_vector_add_vector ( float *d_in1, float *d_in2, float *d_out, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	printf("%s: din1 %p din2 %p out %p tblsize %d\n", __func__,
		d_in1, d_in2, d_out, tbl_size ) ;

	// dbg_p_d_data_f("h_do_vector_add_vector 1", d_in1, tbl_size ) ;
	// dbg_p_d_data_f("h_do_vector_add_vector 2", d_in2, tbl_size ) ;

#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_vector_add_vector <<< nBlocks, nThreadsPerBlock >>> ( d_in1, d_in2, d_out, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_vector_add_vector OUT", d_out, tbl_size ) ;
#endif                                                                           
}

__global__ void d_do_vector_2_norm ( float *d_in, float *d_out, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		d_out[ t_idx ] = d_in[ t_idx ] * d_in[ t_idx ] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

float
h_do_vector_2_norm ( float *d_a, float *d_tmp, int size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	float f ;

#ifdef CUDA_DBG 
	printf("%s: din %p dout %p tblsize %d\n", __func__, d_a, d_tmp, size ) ;

#endif 

	h_block_adj ( size, nThreadsPerBlock, &nBlocks ) ;

	d_do_vector_2_norm <<< nBlocks, nThreadsPerBlock >>> ( d_a, d_tmp, size ) ; 

	cudaThreadSynchronize() ;

	f = h_do_vector_add_destroy ( d_tmp, size ) ;

	f = sqrtf( f ) ;

#ifdef CUDA_DBG 
	printf("%s: 2 norm %f \n", __func__, f ) ;
#endif
	return ( f ) ;
}

__global__ void d_do_max_destroy ( float *in1, int to_idx, int from_idx )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < to_idx )
	{
		if (( t_idx >= from_idx ) && ( t_idx ))
		{
			if ( in1 [ t_idx ] > in1[ t_idx - from_idx ] )
				in1[ t_idx - from_idx ] = in1[ t_idx ] ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

float
h_do_max_destroy ( float *d_in, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int j, i, nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	float f ;

	i = max_log2( tbl_size ) ;

	while ( i != 1 )
	{
#ifdef CUDA_DBG 
		dbg_p_d_data_f("h_do_max_destroy before", d_in, tbl_size ) ;
#endif                                                                           
		j = ( i >> 1 ) ;

		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_max_destroy <<< nBlocks, nThreadsPerBlock >>> ( d_in, tbl_size, j ) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
		dbg_p_d_data_f("h_do_max_destroy after", d_in, tbl_size ) ;
#endif                                                                           
		tbl_size = j ;
		i >>= 1 ;
	}
	get_d_data_f ( d_in, &f, sizeof( f )) ; 
	return ( f ) ;
}

float
h_do_vector_inf_norm ( float *d_a, float *d_tmp, int size )
{
	float f ;

#ifdef CUDA_DBG 
	printf("%s: din %p dout %p tblsize %d\n", __func__, d_a, d_tmp, size ) ;
#endif 

	h_do_copy_vector( d_a, d_tmp, size ) ;

	h_do_abs_vector( d_tmp, size ) ;

	f = h_do_max_destroy ( d_tmp, size ) ;

#ifdef CUDA_DBG 
	printf("%s: inf norm %f \n", __func__, f ) ;
#endif                                                                           
	return ( f ) ;
}	

// MATLAB -> C

__global__ void d_do_vhtc_2_hvtc( float *d_in, float *d_out, int v, int h, int t, int c, int size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j, frame_size, row, col ;

	frame_size = v * h ;

	while ( t_idx < size )
	{
		i = t_idx / frame_size ;
		j = t_idx % frame_size ;

		col = j / v ;
	    row = j % v ;	

		d_out [ i * frame_size + row * h + col ] = d_in [ t_idx ] ;
		
		t_idx += CUDA_MAX_THREADS ;
	}
}

/* h_do_vhtc_2_hvtc:
   	to change the data format from matlab vhtc to c hvtc ... not the order but the size

		so matlab 72x88x2 
		in c 	88x72x2
	
	MATLAB -> C 

NOTE: d_in and d_out can not be the same
*/

void
h_do_vhtc_2_hvtc ( float *d_in, float *d_out, int v, int h, int t, int c )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int i, nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	i = v * h * t * c ;

#ifdef CUDA_DBG 
	// dbg_p_d_data_f_mn("h_do_vhtc_2_hvtc before", d_in, v * h * t * c, h, v, h ) ;
	dbg_p_d_data_f("h_do_vhtc_2_hvtc before", d_in, v * h * t * c ) ;
#endif                                                                           

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_do_vhtc_2_hvtc <<< nBlocks, nThreadsPerBlock >>> ( d_in, d_out, v, h, t, c, i ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn("h_do_vhtc_2_hvtc after", d_out, v * h * t * c, h, v, h ) ;
	dbg_p_d_data_f("h_do_vhtc_2_hvtc after", d_out, v * h * t * c ) ;
#endif                                                                           
}

// C -> MATLAB

__global__ void d_do_hvtc_2_vhtc( float *d_in, float *d_out, int v, int h, int t, int c, int size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j, frame_size, row, col ;

	frame_size = v * h ;

	while ( t_idx < size )
	{
		i = t_idx / frame_size ;
		j = t_idx % frame_size ;

		row = j / h ;
	    col = j % h ;	

		d_out [ i * frame_size + col * v + row ] = d_in [ t_idx ] ;
		
		t_idx += CUDA_MAX_THREADS ;
	}
}

/* h_do_vhtc_2_hvtc:
   	to change the data format from matlab vhtc to c hvtc ... not the order but the size

		so matlab 72x88x2 
		in c 	88x72x2
	
	C --> MATLAB

NOTE: d_in and d_out can not be the same
*/

void
h_do_hvtc_2_vhtc ( float *d_in, float *d_out, int v, int h, int t, int c )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int i, nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	i = v * h * t * c ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f_mn("h_do_hvtc_2_vhtc before", d_in, v * h * t * c, h, v, h ) ;
	dbg_p_d_data_f("h_do_hvtc_2_vhtc before", d_in, v * h * t * c ) ;
#endif                                                                           

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_do_hvtc_2_vhtc <<< nBlocks, nThreadsPerBlock >>> ( d_in, d_out, v, h, t, c, i ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	// dbg_p_d_data_f_mn("h_do_hvtc_2_vhtc after", d_out, v * h * t * c, h, v, h ) ;
	dbg_p_d_data_f("h_do_hvtc_2_vhtc after", d_out, v * h * t * c ) ;
#endif                                                                           
}
