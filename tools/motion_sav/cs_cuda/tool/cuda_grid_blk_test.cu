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

#define CUDA_DBG 

#define MD_DBG_SIZE 100000

static int h_memp[ MD_DBG_SIZE ] ;

static int *d_memp, *d_dbgp = NULL ;

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
	int *dp,
	int to_tbl_size,
	int *dbgp
	)
{
	int i ;

#ifdef CUDA_OBS 
	if (( blockIdx.x == 3 ) &&
		( blockIdx.y == 3 ) && 
		( blockIdx.z == 0 ) && 
		( threadIdx.x == 2 ) &&
		( threadIdx.y == 0 ) &&
		( threadIdx.z == 0 ))
	{
#endif 
		// GOLD for 2-D
		i = blockIdx.y * ( gridDim.x * blockDim.x ) * blockDim.y +
			threadIdx.y * ( gridDim.x * blockDim.x ) +
			blockIdx.x * blockDim.x + 
			threadIdx.x ;	   

		if ( i < to_tbl_size )
		{
			if ( dbgp[i] == 7777 )
				dbgp[i] = 1 ;
			else
				dbgp[i]++ ;
		}

#ifdef CUDA_DBG 
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
#endif 
#ifdef CUDA_OBS 
	}
#endif 
}

void
h_size_test2( int tbl_size )
{
	dim3 grid, block ;

	block.x = 7 ;
	block.y = 5 ;
	block.z = 1 ;

	grid.x = 10 ;
	grid.y = 4 ;
	grid.z = 1 ;

	set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;

	// h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_size_test2 <<< grid, block >>> (
		d_memp,
		tbl_size,	// does not include the 3 indexes 
		d_dbgp
	   	) ;

	cudaThreadSynchronize() ;

	dbg_p_d_data_i("SIZE CKING2", d_dbgp, 1500 ) ; 
}

__device__ int  
d_max_log2( int i )
{
	int k, j ;

	k = ( int )log2(( double ) i ) ;
	j = (int)pow(2.0, k ) ;

	if ( j == i )
		j >>= 1 ;
					
	return ( j ) ;
}

// note elements handled by a block is actually, at the max, (blockDim.x * 2)
__global__ void d_grid_test( int *odp, int block_size, int record_per_blk,
	int record_size, int skip
#ifdef CUDA_DBG 
	, int *dbp 
#endif 
	)
{
	int j, i, element_for_this_cuda_blk, max_elements_for_this_cuda_blk ;
	int todo, dim_x, done_after, t_idx, start, cnt ;
#ifdef CUDA_DBG 
	int dbg = 0 ;
#endif 

	i = blockIdx.x / record_per_blk ; 

	odp += i * block_size ;	// odp points to beginning of the block

	i = blockIdx.x % record_per_blk ;

	odp += i * record_size ; // odp points to the beginning of the record

	i = blockIdx.y ;	// which cuda_blk

	dim_x = blockDim.x ;
	max_elements_for_this_cuda_blk = dim_x * 2 ;
	j = i * max_elements_for_this_cuda_blk ;

	odp += j ; // odp points to the beginning of the chuck of elements
			// handled by this cuda_block

	todo = ( record_size + skip - 1 ) / skip ;	// how many elements need to be done in this record

	element_for_this_cuda_blk = todo - j ;

	if ( element_for_this_cuda_blk > max_elements_for_this_cuda_blk )
		element_for_this_cuda_blk = max_elements_for_this_cuda_blk ;

#ifdef CUDA_DBG 
	if (( blockIdx.x == 0 ) && ( blockIdx.y == 0 ) && ( threadIdx.x == 0 ))
	{
		*dbp++ = block_size ;
	    *dbp++ = record_per_blk;
		*dbp++ = record_size ;
		*dbp++ = todo ;
		*dbp++ = skip ;

		dbg += 11111111 ;
		*dbp++ = dbg ;

		*dbp++ = gridDim.x ;
		*dbp++ = gridDim.y ;
		*dbp++ = gridDim.z ;

		*dbp++ = blockIdx.x ;
		*dbp++ = blockIdx.y ;
		*dbp++ = blockIdx.z ;

		dbg += 11111111 ;
		*dbp++ = dbg ;

		*dbp++ = blockDim.x ;
		*dbp++ = blockDim.y ;
		*dbp++ = blockDim.z ;

		*dbp++ = threadIdx.x ;
		*dbp++ = threadIdx.y ;
		*dbp++ = threadIdx.z ;

		dbg += 11111111 ;
		*dbp++ = dbg ;

		*dbp++ = element_for_this_cuda_blk ;
		*dbp++ = max_elements_for_this_cuda_blk ;
		*dbp++ = j ;
		*dbp++ = i ;
	}
#endif 

	if ( element_for_this_cuda_blk > 0 )
	{
		start = d_max_log2( element_for_this_cuda_blk ) ;
		done_after = start / 2 ;

		cnt = element_for_this_cuda_blk - start ;

		while ( element_for_this_cuda_blk > 1 )
		{

#ifdef CUDA_DBG 
			if (( blockIdx.x == 0 ) && ( blockIdx.y == 0 ) && ( threadIdx.x == 0 ))
			{
				dbg += 11111111 ;
				*dbp++ = dbg ;

				*dbp++ = start ;
				*dbp++ = done_after ;
				*dbp++ = cnt ;
				*dbp++ = element_for_this_cuda_blk ;
			}
#endif 

			t_idx = threadIdx.x ;

			if ( t_idx < cnt )
				odp [ t_idx * skip ] += odp [ ( t_idx + start ) * skip ] ;

			if ( t_idx >= done_after )
				break ; // this thread is done
 
			element_for_this_cuda_blk -= cnt ;

			start >>= 1 ;
			cnt = start ;
			done_after = cnt / 2 ;

			__syncthreads() ;
		}
	}
}

// this is to use the block __syncthread ... grid.y will have the block of a record
void
h_grid_test ()
{
	dim3 blk, grid ;
	int *ip, skip, todo, i, j, block_size, total_record, record_per_blk, record_size, cuda_blk_size ;

	record_size = 65 ;
	block_size = 300 ;

	if ( MD_DBG_SIZE < ( block_size * 2 ))
	{
		printf("%s: error MD_DBG_SIZE %d block_size %d \n", __func__, MD_DBG_SIZE, block_size ) ;
		return ;
	}

	record_per_blk = 2 ;
	total_record = 3 ;

	cuda_blk_size = 32 ;

	ip = h_memp ;
	j = ( total_record + record_per_blk - 1 ) / record_per_blk  ;
	while ( j-- )
	{
		for ( i = 0 ; i < record_size ; i++ )
			*ip++ = 1 ;

		for ( i = 0 ; i < record_size ; i++ )
			*ip++ = 2 ;

		for ( i = 0 ; i < ( block_size - record_per_blk * record_size ) ; i++ )
			*ip++ = 1234567 ;
	}

	put_d_data_i ( d_memp, h_memp, MD_DBG_SIZE ) ;

	// dbg_p_d_data_i("D_MEMP ", d_memp, 500 ) ; 

	set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;

	blk.x = cuda_blk_size ;
	blk.y = blk.z = 1 ;

	grid.x = total_record ;	// totally, 3 records
	grid.y = ( record_size + 2 * cuda_blk_size - 1 ) / ( 2 * cuda_blk_size ) ;
		// number of cuda blocks per records 
	grid.z = 1 ;

	printf("%s : grid %d %d %d blk %d %d %d \n", __func__,
		grid.x,
		grid.y,
		grid.z,
		blk.x,
		blk.y,
		blk.z ) ;

	skip = 1 ;
	todo = record_size ;
	i = record_size ;
	while ( todo > 1 )
	{
		d_grid_test <<< grid, blk >>> ( d_memp, block_size, record_per_blk,
			record_size, skip 
#ifdef CUDA_DBG 
			, d_dbgp
#endif 
			) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
		dbg_p_d_data_i("D_LOOP ", d_memp, 500 ) ; 

		dbg_p_d_data_i("DBG ", d_dbgp, 50 ) ; 
		set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;
#endif 

		// break ; // TTT

		todo = ( todo + 2 * cuda_blk_size - 1 ) / ( 2 * cuda_blk_size ) ; // this much left to do

		skip *= ( 2 * cuda_blk_size ) ;

		grid.y = ( todo + 2 * cuda_blk_size - 1 ) / ( 2 * cuda_blk_size ) ; // 

		printf("loop : skip %d todo %d \n", skip, todo ) ;
	}
	dbg_p_d_data_i("DONE ", d_memp, 500 ) ; 
}

main( int ac, char *av[] )
{
	int nThreadsPerBlock = 512 ;
	int tbl_size = 1000000 ;
	int i, nBlocks ; //=  ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	dbg_init( 3000000 ) ;

	setbuf ( stdout, NULL ) ;
	setbuf ( stderr, NULL ) ;

	if (( nBlocks = cudaMalloc( &d_memp, MD_DBG_SIZE * sizeof( int ))) != cudaSuccess )
	{
		printf("%s: dbg cudaMalloc failed %d\n", __func__, nBlocks ) ;
		exit ( 3 ) ;
	}
	printf("d_memp %p \n", d_memp ) ; 

	for ( i = 0 ; i < MD_DBG_SIZE ; i++ )
		h_memp[i] = i ;

	put_d_data_i ( d_memp, h_memp, MD_DBG_SIZE ) ;

	// dbg_p_d_data_i("D_MEMP ", d_memp, 200 ) ; 

	if (( nBlocks = cudaMalloc( &d_dbgp, 10000 * sizeof( int ))) != cudaSuccess )
	{
		printf("%s: dbg cudaMalloc failed %d\n", __func__, nBlocks ) ;
		exit ( 3 ) ;
	}
	printf("d_dbgp %p \n", d_dbgp ) ; 

	set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;

#ifdef CUDA_OBS 
	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_size_test <<< nBlocks, nThreadsPerBlock >>> (
		tbl_size,	// does not include the 3 indexes 
		d_dbgp
	   	) ;

	cudaThreadSynchronize() ;

	dbg_p_d_data_i("SIZE CKING", d_dbgp, 15 ) ; 
#endif 

#ifdef CUDA_OBS 
	h_size_test2( tbl_size ) ;
#endif 

	h_grid_test() ;

	exit(0);
}
