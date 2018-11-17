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

#include "cs_perm_generic.h"

// #define CUDA_DBG
// #define CUDA_DBG1

__global__ void d_do_permutation_generic_f1 ( int *input, int *output,
	int *idxp, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		output[ idxp[ t_idx ]] = input[ t_idx ] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
	perform

	target(per(i))=orig(i)

	target: outcome vector
	orig: the original vector
	per: permutation vector
*/
void
h_do_permutation_generic_f1 ( int *d_input, int *d_output, int *d_perm_tbl,
	int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p tblsize %d\n", __func__,
		d_input, d_output, d_perm_tbl, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_generic_f1 <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("h_do_permutation_generic_f1 perm before", d_input, tbl_size ) ; 
	dbg_p_d_data_i("h_do_permutation_generic_f1 perm after", d_output, tbl_size ) ; 
#endif 
}

__global__ void d_do_permutation_generic_f2 ( int *input, int *output,
	int *idxp, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		output[t_idx] = input[ idxp[t_idx]] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
	perform

	target(i)=orig((per(i))

	target: outcome vector
	orig: the original vector
	per: permutation vector
*/
void
h_do_permutation_generic_f2 ( int *d_input, int *d_output, int *d_perm_tbl,
	int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p size %d \n", __func__,
		d_input, d_output, d_perm_tbl, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_generic_f2 <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("h_do_permutation_generic_f2 perm before", d_input, tbl_size ) ; 
	dbg_p_d_data_i("h_do_permutation_generic_f2 perm after", d_output, tbl_size ) ; 
#endif 
}

// same logic as above but the data types are float

__global__ void d_do_permutation_generic_f1 ( float *input, float *output,
	int *idxp, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		output[ idxp[ t_idx ]] = input[ t_idx ] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
	perform

	target(per(i))=orig(i)

	target: outcome vector
	orig: the original vector
	per: permutation vector
*/
void
h_do_permutation_generic_f1 ( float *d_input, float *d_output, int *d_perm_tbl,
	int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p tblsize %d\n", __func__,
		d_input, d_output, d_perm_tbl, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_generic_f1 <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_permutation_generic_f1 perm before", d_input, tbl_size ) ; 
	dbg_p_d_data_f("h_do_permutation_generic_f1 perm after", d_output, tbl_size ) ; 
#endif 
}

__global__ void d_do_permutation_generic_f2 ( float *input, float *output,
	int *idxp, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		output[t_idx] = input[ idxp[t_idx]] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
	perform

	target(i)=orig((per(i))

	target: outcome vector
	orig: the original vector
	per: permutation vector
*/
void
h_do_permutation_generic_f2 ( float *d_input, float *d_output, int *d_perm_tbl,
	int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p size %d tblsize %d\n", __func__,
		d_input, d_output, d_perm_tbl, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_generic_f2 <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_permutation_generic_f2 perm before", d_input, tbl_size ) ; 
	dbg_p_d_data_f("h_do_permutation_generic_f2 perm after", d_output, tbl_size ) ; 
#endif 
}

__global__ void d_do_permutation_generic_inverse ( int *output,
	int *idxp, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		output[ idxp[ t_idx ]] = t_idx ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
	perform

	target(per(i))=(i)

	target: outcome vector
	orig: the original vector
	per: permutation vector
*/
void
h_do_permutation_generic_inverse ( int *d_output, int *d_perm_tbl,
	int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: dout %p perm %p tblsize %d\n", __func__,
		d_output, d_perm_tbl, tbl_size ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_generic_inverse <<< nBlocks, nThreadsPerBlock >>> (
		d_output, d_perm_tbl, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("h_do_permutation_generic_inverse perm before", d_perm_tbl, tbl_size ) ; 
	dbg_p_d_data_i("h_do_permutation_generic_inverse perm after", d_output, tbl_size ) ; 
#endif 
}
