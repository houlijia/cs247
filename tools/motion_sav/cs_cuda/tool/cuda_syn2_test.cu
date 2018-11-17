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

	odp += j * skip ; // odp points to the beginning of the chuck of elements
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

		*dbp++ = j * skip ;

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

	record_size = 600 ;
	block_size = 1500 ;

	if ( MD_DBG_SIZE < ( block_size * 2 ))
	{
		printf("%s: error MD_DBG_SIZE %d block_size %d \n", __func__, MD_DBG_SIZE, block_size ) ;
		return ;
	}

	record_per_blk = 2 ;
	total_record = 3 ;

	cuda_blk_size = 16 ;

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

	skip = 1 ;
	todo = record_size ;
	i = record_size ;
	while ( todo > 1 )
	{
		printf("%s : todo %d ==  grid %d %d %d blk %d %d %d \n", __func__,
			todo, 
			grid.x,
			grid.y,
			grid.z,
			blk.x,
			blk.y,
			blk.z ) ;

		d_grid_test <<< grid, blk >>> ( d_memp, block_size, record_per_blk,
			record_size, skip 
#ifdef CUDA_DBG 
			, d_dbgp
#endif 
			) ;

		cudaThreadSynchronize() ;

		dbg_p_d_data_i("D_LOOP ", d_memp, 5000 ) ; 

		dbg_p_d_data_i("DBG ", d_dbgp, 50 ) ; 
		set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;

		// break ; // TTT

		todo = ( todo + 2 * cuda_blk_size - 1 ) / ( 2 * cuda_blk_size ) ; // this much left to do

		skip *= ( 2 * cuda_blk_size ) ;

		grid.y = ( todo + 2 * cuda_blk_size - 1 ) / ( 2 * cuda_blk_size ) ; // 

		printf("LOOP loop : skip %d todo %d \n", skip, todo ) ;
	}
	dbg_p_d_data_i("DONE ", d_memp, 5000 ) ; 
}

main( int ac, char *av[] )
{
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

	h_grid_test() ;

	exit(0);
}
