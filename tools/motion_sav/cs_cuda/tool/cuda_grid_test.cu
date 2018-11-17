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
#include "cs_helper.h"
#include "cs_video_io.h"
#include "cs_config.h"

#define MD_DBG_SIZE 1024

static int *d_dbgp = NULL ;

__global__ void d_size_test (
	int to_tbl_size,
	int *dbgp
	)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ( t_idx == 27 )
	{
		*dbgp++ = t_idx ;

		*dbgp++ = gridDim.x ;
		*dbgp++ = gridDim.y ;
		*dbgp++ = gridDim.z ;

		// *dbgp++ = gridIdx.x ;
		// *dbgp++ = gridIdx.y ;
		// *dbgp++ = gridIdx.z ;

		*dbgp++ = blockDim.x ;
		*dbgp++ = blockDim.y ;
		*dbgp++ = blockDim.z ;

		*dbgp++ = blockIdx.x ;
		*dbgp++ = blockIdx.y ;
		*dbgp++ = blockIdx.z ;

		*dbgp++ = threadIdx.x ;
		*dbgp++ = threadIdx.y ;
		*dbgp++ = threadIdx.z ;

		*dbgp++ = t_idx ;
	}
}

__global__ void d_size_test2 (
	int to_tbl_size,
	int *dbgp
	)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	*dbgp++ = 333 ;

	if ( t_idx == 0 )
	{
		*dbgp++ = t_idx ;

		*dbgp++ = gridDim.x ;
		*dbgp++ = gridDim.y ;
		*dbgp++ = gridDim.z ;

		*dbgp++ = blockDim.x ;
		*dbgp++ = blockDim.y ;
		*dbgp++ = blockDim.z ;

		*dbgp++ = blockIdx.x ;
		*dbgp++ = blockIdx.y ;
		*dbgp++ = blockIdx.z ;

		*dbgp++ = threadIdx.x ;
		*dbgp++ = threadIdx.y ;
		*dbgp++ = threadIdx.z ;

		*dbgp++ = t_idx ;
	}
}

void
h_size_test2( int tbl_size )
{
	dim3 grid, block ;

	block.x = 1 ;
	block.y = 1 ;
	block.z = 1 ;

	grid.x = 1 ;
	grid.y = 1 ;
	grid.z = 1 ;

	printf("in test2 \n") ;

	set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;

	// h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_size_test2 <<< grid, block >>> (
		tbl_size,	// does not include the 3 indexes 
		d_dbgp
	   	) ;

	cudaThreadSynchronize() ;

	dbg_p_d_data_i("SIZE CKING2", d_dbgp, 20 ) ; 
}

main( int ac, char *av[] )
{
	int nThreadsPerBlock = 512 ;
	int tbl_size = 1000000 ;
	int nBlocks ; //=  ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	dbg_init( 3000 ) ;

	if (( nBlocks = cudaMalloc( &d_dbgp, MD_DBG_SIZE * sizeof( int ))) != cudaSuccess )
	{
		printf("%s: dbg cudaMalloc failed %d\n", __func__, nBlocks ) ;
		exit ( 3 ) ;
	}
	printf("d_dbgp %p \n", d_dbgp ) ; 

#ifdef CUDA_OBS 
	set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_size_test <<< nBlocks, nThreadsPerBlock >>> (
		tbl_size,	// does not include the 3 indexes 
		d_dbgp
	   	) ;

	cudaThreadSynchronize() ;

	dbg_p_d_data_i("SIZE CKING", d_dbgp, 15 ) ; 
#endif 

	h_size_test2( tbl_size ) ;

	exit(0);
}
