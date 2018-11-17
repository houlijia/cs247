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

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_header.h"
#include "cs_block.h"
#include "cs_perm_mlseq.h"
#include "cs_expand.h"
#include "cs_interpolate.h"
#include "cs_perm_selection.h"
#include "cs_copy_box.h"
#include "cs_motion_detect.h"
#include "cs_motion_detect_v2.h"
#include "cs_edge_detect.h"
#include "cs_edge_detect_v2.h"
#include "cs_analysis.h"

#define CUDA_DBG

int *dp1 = NULL, *dp2 = NULL ;
int *hp1 = NULL, *hp2 = NULL ;

// #define BUF_SIZE	( 1024 * 1024 * 32 )
#define MAGIC_SIZE 33553920 
#define BUF_SIZE	( MAGIC_SIZE + 100 )

struct cd {
	int tid ;

	// gridDim
	int gdx ;
	int gdy ;
	int gdz ;

	// blockIdx
	int blkx ;
	int blky ;
	int blkz ;
	
	// blockDim
	int dx ;
	int dy ;
	int dz ;

	// threadIdx
	int tx ;
	int ty ;
	int tz ;

	int cnt ;
} ;

#define CD_SIZE_E	(sizeof( struct cd ) / sizeof(int))

enum {
	T2DB_1DG = 1,
	T3DB_1DG,
	T3DB_2DG,
	T3DB_3DG,
	T1DB_1DG
} ;

void cuda_test_grid ( struct cd *d_a, int n )	 ;

// #define TEST_LEN		128 // AAA 2D blocks ... 1D grid
#define TEST_LEN		576	// BBB 3D blocks ... 1D grid

__global__ void d_cuda_test_grid( struct cd *a, int size, int add, int ttt )
{
	int blockid, tid ;

	switch ( ttt ) {
	case T1DB_1DG :
		tid = blockIdx.x * blockDim.x + threadIdx.x ; 
		break ;

	case T2DB_1DG :
		tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x ; // AAA
		break ;

	case T3DB_1DG :
		tid = blockIdx.x * ( blockDim.x * blockDim.y * blockDim.z ) + 
			threadIdx.z * ( blockDim.x * blockDim.y ) + 
			threadIdx.y * blockDim.x + threadIdx.x ; // BBB 
		break ;
	
	case T3DB_3DG :

		// NOTE: gridDim.z is never used in the calculation, but is used
		// by the engine for size limitation ... pass this limit, the
		// engine stop ...
		blockid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z *
			( gridDim.x * gridDim.y ) ; 

		tid = blockid * ( blockDim.x * blockDim.y * blockDim.z ) +
			threadIdx.z * ( blockDim.x * blockDim.y ) + 
			threadIdx.y * blockDim.x + threadIdx.x ;

		break ;

	case T3DB_2DG :
		blockid = blockIdx.x + blockIdx.y * gridDim.x ; 
		tid = blockid * ( blockDim.x * blockDim.y * blockDim.z ) +
			threadIdx.z * ( blockDim.x * blockDim.y ) + 
			threadIdx.y * blockDim.x + threadIdx.x ;
#ifdef CUDA_OBS 
		tid = blockIdx.x * gridDim.x * ( blockDim.x * blockDim.y * blockDim.z ) + 
			blockIdx.x * ( blockDim.x * blockDim.y * blockDim.z ) + 
			threadIdx.z * ( blockDim.x * blockDim.y ) + 
			threadIdx.y * blockDim.x + threadIdx.x ; // BBB 
#endif 
		break ;
	
	default :
		tid = blockIdx.x * ( blockDim.x * blockDim.y * blockDim.z ) + 
			threadIdx.z * ( blockDim.x * blockDim.y ) + 
			threadIdx.y * blockDim.x + threadIdx.x ; // BBB 
	}

	while ( tid < size )
	{
		a[tid].tid = tid ;

		a[tid].blkx = blockIdx.x ;
		a[tid].blky = blockIdx.y ;
		a[tid].blkz = blockIdx.z ;

		a[tid].tx = threadIdx.x ;
		a[tid].ty = threadIdx.y ; 
		a[tid].tz = threadIdx.z ; 

		a[tid].gdx = gridDim.x ; 
		a[tid].gdy = gridDim.y ; 
		a[tid].gdz = gridDim.z ;

		a[tid].dx = blockDim.x ; 
		a[tid].dy = blockDim.y ; 
		a[tid].dz = blockDim.z ;

		a[tid].cnt++ ;

		tid += size ;
	} 
}

__global__ void d_cuda_test_inc ( struct cd *a, int size, int add, int ttt )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x ; 

	while ( tid < size )
	{
		a[tid].tid = tid ;

		a[tid].blkx = blockIdx.x ;
		a[tid].blky = blockIdx.y ;
		a[tid].blkz = blockIdx.z ;

		a[tid].cnt++ ;

		tid += add ;
	}
}

void
cuda_test_inc ( struct cd *d_a, int n, int ttt )	
{

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: dp %p cnt %d type %d\n", __func__, d_a, n, ttt ) ;
#endif 

	d_cuda_test_inc <<< 2, 32 >>> (d_a, n, 64, ttt ) ; 

	cudaThreadSynchronize() ;
}

void
cuda_test_grid ( struct cd *d_a, int n, int ttt )	
{
	dim3 threadsPerBlock(8,8,1) ; 
	dim3 nBlocks( 2, 2 ) ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: dp %p cnt %d type %d\n", __func__, d_a, n, ttt ) ;
#endif 

	switch ( ttt ) {
	case T1DB_1DG :
		d_cuda_test_grid <<< 2, 32 >>> (d_a, n, n, ttt ) ; 
		break ;

	case T2DB_1DG :
		threadsPerBlock.x = 8 ;
		threadsPerBlock.y = 8 ;
		threadsPerBlock.z = 1 ;
		d_cuda_test_grid <<< 2, threadsPerBlock >>> (d_a, n, n, ttt ) ; 
		break ;

	case T3DB_1DG:
		threadsPerBlock.x = 4 ;
		threadsPerBlock.y = 3 ;
		threadsPerBlock.z = 2 ;
		d_cuda_test_grid <<< 2, threadsPerBlock >>> (d_a, n, n, ttt ) ;
		break ;

	case T3DB_2DG:

		threadsPerBlock.x = 4 ;
		threadsPerBlock.y = 3 ;
		threadsPerBlock.z = 2 ;

		nBlocks.x = 3 ; // ( n / threadsPerBlock.x, n / threadsPerBlock.y ) ;
		nBlocks.y = 2 ; 

		d_cuda_test_grid <<< nBlocks, threadsPerBlock >>> (d_a, n, n, ttt ) ;
		break ;

	case T3DB_3DG:

		threadsPerBlock.x = 4 ;
		threadsPerBlock.y = 3 ;
		threadsPerBlock.z = 2 ;

		nBlocks.x = 3 ;
		nBlocks.y = 2 ; 
		nBlocks.z = 4 ; 

		d_cuda_test_grid <<< nBlocks, threadsPerBlock >>> (d_a, n, n, ttt ) ;
		break ;

	default :
		exit( 33 ) ;
	}

	cudaThreadSynchronize() ;
}

main( int ac, char *av[] )
{
	int i, k, *dp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	i = BUF_SIZE * sizeof ( struct cd ) ;

	if (( k = cudaMalloc( &dp, i )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	printf("size %d\n", i ) ;

	set_device_mem_i ( dp, TEST_LEN * CD_SIZE_E * sizeof ( int ) + 200, 1 ) ; 

	dbg_init( 1024 * 1024 ) ;

	// test of increments of tid in each thread

	cuda_test_inc (( struct cd *) dp, TEST_LEN, T1DB_1DG ) ;	
	dbg_p_d_data_i_mn ("done", dp, CD_SIZE_E * TEST_LEN, CD_SIZE_E, TEST_LEN, CD_SIZE_E ) ;

#ifdef CUDA_OBS 

	// test of thread/block/grid ...
	cuda_test_grid (( struct cd *) dp, TEST_LEN, T1DB_1DG ) ;	
	dbg_p_d_data_i_mn ("done", dp, CD_SIZE_E * TEST_LEN, CD_SIZE_E, TEST_LEN, CD_SIZE_E ) ;

#endif 
}
