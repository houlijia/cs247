#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_header.h"
#include "cs_dbg.h"
#include "cs_analysis.h"
#include "cs_helper.h"
#include "cs_cuda.h"
#include "cs_motion_detect_v4.h"

/*
Fri Jun  5 10:56:53 EDT 2015

v3 is based on per block, too slow.  v4 will change to work on the whole video-frames at
a time.  i.e. manipulate the data on a per whole-frames basis.

Fri Apr 24 17:46:50 EDT 2015

version 2 will only match the first md block with the subsequent md_z -1 block to do
the L1 norm calculation.  which works fine, but not too accurate, v3 will change that
to compare the "center" md blocks across the termperal domain
*/

// step two is to get L1-norm(sum)
// all row, should be after the abs() is done
// tbl_size is the number of elements for this addition operation
// entries_in_block: max number of entries to work
// blk_size: is the max of the diff types
// cube.z is the starting entry, cube.y is the size for this record
// cnt is the max_cnt for each record, regardless inner/side/corner

template<typename T>
__global__ void d_do_l1_norm_step2_v4 ( T *odp, int tbl_size, int entries_in_blk,
	struct cube *d_cubep, int blk_size, int blk_in_x, int blk_in_y, int hvt_adj
#ifdef CUDA_DBG 
	, int *d_dbgp	
#endif 
	)
{
	int ot_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx, start, j ;
	int loopcnt, blk, record_work_size, blk_type_idx ;
	T *fp, *dp ;

	t_idx = ot_idx ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		blk = t_idx / entries_in_blk ;	// j is the block index

		blk_type_idx = get_blk_type_idx( blk, blk_in_x, blk_in_y ) ;

		t_idx -= blk * entries_in_blk ; // t_idx, element idx in this block

		record_work_size = d_cubep[ blk_type_idx ].y ; // size
		loopcnt = d_cubep[ blk_type_idx ].md_v4_loopcnt ;

		j = record_work_size * loopcnt ;

		if ( t_idx < j )
		{
			dp += blk * blk_size ;	// dp points to block ...

			start = d_cubep[ blk_type_idx ].z ; 

			j = t_idx / record_work_size ;

			dp += j * d_cubep[ blk_type_idx ].md_v4_record_length ; // dp points to start of record

			t_idx -= j * record_work_size ; // t_idx, index to the element of the record

			dp += hvt_adj ;	// dp points to the start of the elements
			fp = dp + start ;

#ifdef CUDA_OBS 
				// this check is needed when *dp is int
				{
					l = dp[ j ] ;
					ll = fp [ j ] ;

					l += ll ;
		
					if ( l & 0xffffffff00000000 )
						*d_resp = t_idx ;
				}
#endif 

			dp[ t_idx ] += fp [ t_idx ] ;
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}

template<typename T>
__global__ void d_do_l1_norm_step2_v4_block_v1( T *odp, struct cube *d_cubep,
	int block_size, 
	int blk_in_x, int blk_in_y, int skip
#ifdef CUDA_OBS 
	, int *dbp 
#endif 
	)
{
	int j, i, element_for_this_cuda_blk, max_elements_for_this_cuda_blk ;
	int todo, dim_x, done_after, t_idx, start, cnt ;
	int record_per_blk, record_size ;
#ifdef CUDA_OBS 
	int dbg = 0 ;
#endif 

	i = blockIdx.z ;	// i is blk_index 

	odp += i * block_size ;	// odp points to beginning of the block

	i = get_blk_type_idx( i, blk_in_x, blk_in_y ) ;

	record_size = d_cubep[ i ].md_v4_record_length ;
	record_per_blk = d_cubep[ i ].md_v4_loopcnt ;

	i = blockIdx.x ;

	if ( i >= record_per_blk )
		return ;

	odp += i * record_size ; // odp points to the beginning of the record

	odp += NUM_OF_HVT_INDEX ;

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

#ifdef CUDA_OBS 
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

#ifdef CUDA_OBS 
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

// step two is to get L1-norm(sum)
	// all row, should be after the abs() is done
	// tbl_size is the number of elements for this addition operation
	// entries_in_block: max number of entries to work
	// blk_size: is the max of the diff types
	// cube.z is the starting entry, cube.y is the size for this record
	// cnt is the max_cnt for each record, regardless inner/side/corner

// d_do_l1_norm_step2_v4_block is basically the same as d_do_l1_norm_step2_v4
// but will use the per block sync approach ...

template<typename T>
__global__ void d_do_l1_norm_step2_v4_block ( T *odp, struct cube *d_cubep, 
	int max_row, int blk_size, int blk_in_x, int blk_in_y
#ifdef CUDA_DBG 
	, int *d_dbgp	
#endif 
	)
{
	int total_size, i, k, start ;
	int t_per_block, tidx, last_chunk, blk, blk_type_idx ;
	T *fp, *tp ;

	tidx = blockIdx.y * ( gridDim.x * blockDim.x ) * blockDim.y +
		threadIdx.y * ( gridDim.x * blockDim.x ) +
		blockIdx.x * blockDim.x +
		threadIdx.x ;

	t_per_block = max_row * blockDim.x ;

	i = tidx ; // debug

	blk = tidx / t_per_block ;	// block number 
	blk_type_idx = get_blk_type_idx( blk, blk_in_x, blk_in_y ) ;

	tidx = tidx % t_per_block ; // inside this block
	tidx /= blockDim.x ;	// which record, i.e. cuda block

#ifdef CUDA_OBS 
	if (( blockIdx.x == 1 ) && ( threadIdx.x == 1 ))
	{
		*d_dbgp++ = i ;
		*d_dbgp++ = tidx ;
		*d_dbgp++ = max_row ;
		*d_dbgp++ = blk ;
		*d_dbgp++ = blk_type_idx ;
		*d_dbgp++ = t_per_block ;
		*d_dbgp++ = 8888 ;

		*d_dbgp++ = threadIdx.x ;
		*d_dbgp++ = threadIdx.y ;
		*d_dbgp++ = threadIdx.z ;

		*d_dbgp++ = blockIdx.x ;
		*d_dbgp++ = blockIdx.y ;
		*d_dbgp++ = blockIdx.z ;

		*d_dbgp++ = gridDim.x ;
		*d_dbgp++ = gridDim.y ;
		*d_dbgp++ = gridDim.z ;

		*d_dbgp++ = blockDim.x ;
		*d_dbgp++ = blockDim.y ;
		*d_dbgp++ = blockDim.z ;
	}
#endif 

	if ( tidx < d_cubep[ blk_type_idx ].md_v4_loopcnt )
	{
		odp += blk_size * blk ;	// odp points to the begining of the block
		odp += ( tidx * d_cubep[ blk_type_idx ].md_v4_record_length ) + NUM_OF_HVT_INDEX ;
		// odp points to the data part of the record that this block of threads should handle

		total_size = d_cubep[ blk_type_idx ].md_v4_record_length - NUM_OF_HVT_INDEX ;

		blk_size = blockDim.x ;

		k = ( total_size + blk_size - 1 ) / blk_size ;
		last_chunk = total_size % blk_size ;

		if ( last_chunk )
			k-- ;

		for ( i = 1 ; i <= k ; i++ )
		{
			tp = odp ;
			fp = tp + i * blk_size ;        // first chunk
			if ( i < k )
				tp[ threadIdx.x ] += fp [ threadIdx.x ] ;
			else if ( threadIdx.x < last_chunk )
				tp[ threadIdx.x ] += fp [ threadIdx.x ] ;

			__syncthreads() ; // wait for all threads in this block
		}

		if ( total_size > blk_size )
			total_size = blk_size ;

		start = blk_size / 2 ;  // either 1/2 of DATA_MAX ( if DATA_MAX > BLK_DIM )
		// or max.mod_log2 of data ... ex. if data is 31, it is 16
		last_chunk = total_size - start ;

		while ( total_size > 1 )
		{
			if ( last_chunk > 0 )
			{
				tp = odp ;
				fp = odp + start ;

				if ( threadIdx.x < last_chunk )
					tp[ threadIdx.x ] += fp [ threadIdx.x ] ;
				else
					return ;
			}

			__syncthreads() ; // wait for all threads in this block

			if ( last_chunk > 0 )
				total_size -= last_chunk ;

			start >>= 1 ;
			last_chunk = total_size - start ;
		}
	}
}

// note elements handled by a block is actually, at the max, (blockDim.x * 2)
// cf. d_grid_test@cuda_grid_blk_test.cu
// record_per_blk is the max one ...
// 
template<typename T>
__global__ void d_do_l1_norm_step2_v4_block_v2( T *odp, struct cube *d_cubep,
	int block_size, 
	int blk_in_x, int blk_in_y, int skip,
	int blk_type_idx
#ifdef CUDA_OBS 
	, int *dbp 
#endif 
	)
{
	int j, i, element_for_this_cuda_blk, max_elements_for_this_cuda_blk ;
	int todo, dim_x, done_after, t_idx, start, cnt ;
	int record_per_blk, record_size ;
#ifdef CUDA_OBS 
	int do_dbg = 0 ;
	int dbg = 0 ;
#endif 

	j = blockIdx.z ;	// i is blk_index 

#ifdef CUDA_OBS 
	if (( blk_type_idx == CUBE_INFO_INNER ) && ( j == 3 ) && ( threadIdx.x == 0 ) &&
		( blockIdx.y == 0 ))
	{
		*dbp++ = j ;
		do_dbg++ ;
	} 
#endif 

	switch ( blk_type_idx ) {
	case CUBE_INFO_INNER :
		i = (( j / (blk_in_x - 2 )) + 1 ) * blk_in_x + (j % ( blk_in_x - 2 )) + 1 ; 

		break ;

	case CUBE_INFO_SIDE :
		if ( j < ( blk_in_x - 2 ))
			i = j + 1 ;	// first row
		else
		{
			j = ( j - ( blk_in_x - 2 )) ;

			if ( j >= ( 2 * ( blk_in_y - 2 )))
			{
				// last row
				j -= ( 2 * ( blk_in_y - 2 )) ; 
				i = blk_in_x * ( blk_in_y - 1 ) + j + 1 ;
			} else
			{
				i = blk_in_x + blk_in_x * ( j / 2 ) ;
				j %= 2 ;
				if ( j )
					i += ( blk_in_x - 1 ) ;
			}
		}

		break ;

	case CUBE_INFO_CORNER :
		if ( j & 1 )
		{
			if ( j & 2 )
				i = blk_in_x * blk_in_y - 1 ;
			else
				i = blk_in_x - 1 ;
		} else
		{
			if ( j & 2 )
				i = blk_in_x * ( blk_in_y - 1 ) ;
			else
				i = 0 ;
		}
		break ;

	default:
		return ;
	}

#ifdef CUDA_OBS
	if ( do_dbg )
	{
		*dbp++ = i ;
		*dbp++ = j ;
		dbg += 11111111 ;
		*dbp++ = dbg ;
	}
#endif 

	odp += i * block_size ;	// odp points to beginning of the block

	record_size = d_cubep[ blk_type_idx ].md_v4_record_length ;
	record_per_blk = d_cubep[ blk_type_idx ].md_v4_loopcnt ;

	i = blockIdx.x ;

	if ( i >= record_per_blk )
		return ;

	odp += i * record_size ; // odp points to the beginning of the record

	odp += NUM_OF_HVT_INDEX ;

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

#ifdef CUDA_OBS 
	if ( do_dbg )
	{
		*dbp++ = record_size ;
		*dbp++ = record_per_blk ;
		*dbp++ = i ;
		*dbp++ = dim_x ;
		*dbp++ = max_elements_for_this_cuda_blk ;
		*dbp++ = j ;
		*dbp++ = todo ;
		*dbp++ = element_for_this_cuda_blk ;
		*dbp++ = skip ;

		dbg += 11111111 ;
		*dbp++ = dbg ;
	}
#endif 

	if ( element_for_this_cuda_blk > 0 )
	{
		start = d_max_log2( element_for_this_cuda_blk ) ;
		done_after = start / 2 ;

		cnt = element_for_this_cuda_blk - start ;

#ifdef CUDA_OBS 
		if ( do_dbg )
		{
			*dbp++ = start ;
			*dbp++ = done_after ;
			*dbp++ = cnt ;

			dbg += 11111111 ;
			*dbp++ = dbg ;
		}
#endif 

		while ( element_for_this_cuda_blk > 1 )
		{

#ifdef CUDA_OBS 
			if ( do_dbg )
			{
				*dbp++ = start ;
				*dbp++ = done_after ;
				*dbp++ = cnt ;

				dbg += 11111111 ;
				*dbp++ = dbg ;
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

template<typename T>
int
h_do_l1_norm_step2_v4_block( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size )
{
	int j, i, max_elements, max_row;
	dim3 blk, grid ;

	max_row = 0 ;
	max_elements = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		if ( hcubep[i].md_v4_loopcnt > max_row )
			max_row = hcubep[i].md_v4_loopcnt ;

		if ( hcubep[i].md_v4_record_length > max_elements )
			max_elements = hcubep[i].md_v4_record_length ;
	}

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: dp %p blk x/y %d %d blk_size %d max %d dbgp %p max_ele %d\n",
		__func__, dp, blk_in_x, blk_in_y, blk_size, max_row, d_dbgp, max_elements ) ;
#endif 

#ifdef CUDA_OBS 
	set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;
#endif 

	max_elements = find_thread_blk( max_elements ) ;

	blk.x = max_elements ;
	// blk.x = CUDA_MAX_THREADS_P_BLK ;
	blk.y = 1 ;
	blk.z = 1 ;

	i = max_row * blk_in_x * blk_in_y ;

	j = 1 ;
	while ( i > CUDA_MAX_BLKS ) 
	{
		i++ ;
		i >>=1 ; 
		j <<=1 ;
	}
	
	grid.x = i ; // max_row * blk_in_x * blk_in_y ;
	grid.y = j ;
	grid.z = 1 ;

	printf("%s ::: total %d grid x/y/z %d %d %d blk.x %d\n", __func__, max_row * blk_in_x * blk_in_y,
		grid.x, grid.y, grid.z, blk.x ) ;

	d_do_l1_norm_step2_v4_block<T><<< grid, blk >>> ( dp,
		d_cubep, max_row, blk_size, blk_in_x, blk_in_y
#ifdef CUDA_DBG 
	    , d_dbgp	
#endif 
	) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("AFTER step 2", d_dbgp, 30 ) ; 
#endif 

	return ( 1 ) ;
}

// this is to use the block __syncthread ... grid.y will have the block of a record
template<typename T>
int
h_do_l1_norm_step2_v4_block_v2( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size )
{
	// cf h_grid_test@cuda_grid_blk_test.cu

	dim3 blk, grid ;
	int j, skip, todo, record_size ;
	int othreads_per_cuda_blk, threads_per_cuda_blk ;

	// blk.x = CUDA_MAX_THREADS_P_BLK ;
	blk.y = blk.z = 1 ;

	for ( j = 0 ; j < CUBE_INFO_CNT ; j++ )
	{
		record_size = hcubep[j].md_v4_record_length ;

		othreads_per_cuda_blk = threads_per_cuda_blk = find_thread_blk ( record_size ) ;

		blk.x = threads_per_cuda_blk ;

		grid.x = hcubep[j].md_v4_loopcnt ;	// max number of records in block
		grid.y = ( record_size + 2 * threads_per_cuda_blk - 1 ) /
			( 2 * threads_per_cuda_blk ) ; // number of cuda blocks per records

		// grid.z ... number of blocks to handle
		switch ( j ) {
		case CUBE_INFO_INNER :
			grid.z = ( blk_in_x - 2 ) * ( blk_in_y - 2 ) ;
			break ;

		case CUBE_INFO_SIDE :
			grid.z = ( blk_in_x - 2 ) * 2 + 2 * ( blk_in_y - 2 ) ;
			break ;

		case CUBE_INFO_CORNER :
			grid.z = 4 ;
			break ;
		}

		skip = 1 ;
		todo = record_size ;
		while ( todo > 1 )
		{
#ifdef CUDA_OBS 
			printf("%s : record_size %d grid %d %d %d blk %d %d %d todo %d skip %d\n", __func__,
				record_size,
				grid.x,
				grid.y,
				grid.z,
				blk.x,
				blk.y,
				blk.z,
			    todo,
				skip ) ;

			set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 7777 ) ;
#endif 
			d_do_l1_norm_step2_v4_block_v2<T><<<grid, blk>>> ( dp, d_cubep,
				blk_size, blk_in_x, blk_in_y, skip, j
#ifdef CUDA_OBS 
				, d_dbgp
#endif
				) ;

			cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
			dbg_p_d_data_i("LOOP", d_dbgp, 30 ) ; 
			printf("here todo %d done \n", todo) ;
#endif 

			todo = ( todo + 2 * threads_per_cuda_blk - 1 ) / ( 2 * threads_per_cuda_blk ) ;
				// this much left to do

			if ( todo > 1 ) // we are not done
			{
				skip *= ( 2 * othreads_per_cuda_blk ) ;

				threads_per_cuda_blk = find_thread_blk ( todo ) ;

				blk.x = threads_per_cuda_blk;

				grid.y = ( todo + 2 * threads_per_cuda_blk - 1 ) / ( 2 * threads_per_cuda_blk ) ; // 

#ifdef CUDA_OBS 
				printf("crash: threads_per_cuda_blk %d todo %d \n", threads_per_cuda_blk, todo ) ;
				printf("loop : skip %d todo %d grid.y %d blk.x %d\n", skip, todo, grid.y, blk.x ) ;
#endif 
			} else
				break ;
		}
	}

	return ( 1 ) ;
}

// step 1.1 should be the abs() ... not needed, done in step 1 
// step 2 is to do the sum
// NOTE d_cubep->y/z will be destroyed ... x
// hcubep: has been adjusted to the after md_x/y/z size ; 
template<typename T>
int
h_do_l1_norm_step2_v4( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size, int hvt_adj )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks, i ;
	int max_cnt ;
	struct cube cxyz[ CUBE_INFO_CNT ] ;

	memcpy ( cxyz, hcubep, sizeof ( *hcubep ) * CUBE_INFO_CNT ) ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: dp %p blk x/y %d %d\n",
		__func__, dp, blk_in_x, blk_in_y ) ;
#endif 

	max_cnt = set_up_cube_log ( cxyz, hvt_adj ) ;

#ifdef CUDA_OBS 
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		printf(" cxyz %d == size %d loop %d leng %d \n",
			i,
			cxyz[i].size,
			cxyz[i].md_v4_loopcnt,
			cxyz[i].md_v4_record_length ) ;
	}
#endif 

	h_set_cube_config ( d_cubep, cxyz ) ; 

	while ( max_cnt > 0 ) 
	{
		i = blk_in_x * blk_in_y * max_cnt ;

#ifdef CUDA_OBS 
		fprintf( stderr, "cnt %d i %d \n", max_cnt, i ) ;
#endif 
		
		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step2_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
			dp, i, max_cnt,
			d_cubep,
			blk_size,
			blk_in_x, blk_in_y,
			hvt_adj
#ifdef CUDA_DBG 
			, d_dbgp
#endif 
			) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_i("AFTER step 2", d_dbgp, 13 ) ; 
		set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 0 ) ;
#endif 

		cudaThreadSynchronize() ;

		max_cnt = set_up_cube_log_cont ( cxyz ) ;

		h_set_cube_config ( d_cubep, cxyz ) ; 
	}

	h_set_cube_config ( d_cubep, hcubep ) ; 

	return ( 1 ) ;
}

#define MD_DBG_SIZE 1024

#define CUDA_DBG

static int *d_dbgp = NULL ;

__device__ int
get_blk_type_idx ( int blk_idx, int blk_in_x, int blk_in_y )
{
	int i, blk, k ;

	i = blk_idx / blk_in_x ;
	k = blk_idx % blk_in_x ;

	if (( i == 0 ) || ( i == ( blk_in_y - 1 )))
	{
		if (( k == 0 ) || ( k == ( blk_in_x - 1 )))
			blk = CUBE_INFO_CORNER ;
		else
			blk = CUBE_INFO_SIDE ;
	} else
	{
		if (( k == 0 ) || ( k == ( blk_in_x - 1 )))
			blk = CUBE_INFO_SIDE ;
		else
			blk = CUBE_INFO_INNER ;
	}

	return ( blk ) ;
}

// md_x/y/z: total size ... so it is md_x'*2, md_y'*2 and md_z
// total_to_tbl_size: overall entries size, ( innerblock.recordlength -tvh) * innerblock.loopcnt * nblk_x * nblk_y
// to_blk_size: (innerblock.recordlength) * innerblock.loopcnt ... with tvh
// to_blk_entries_size: ( innerblock.md_v4_loopcnt * ( innerblock.md_v4_record_length - tvh )
// from_blk_size,	// the size of the input inner block ... after edge
template<typename T>
__global__ void d_do_motion_detection_step0_v4 (
	T *fdp, T *tdp, 
	int to_tbl_size,
	int to_blk_entries_size,	// max number of elements in a block ... some are not used ...
	int to_blk_size,	// block size ... including tvh and space not used
	int md_x, int md_y, int md_z,
	struct cube *dcubep,
	int from_blk_size,
   	int blk_in_x, int blk_in_y
#ifdef CUDA_DBG 
	, int *dbgp
#endif 
	)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int record_size, mx, mxy_size ;
	int ot_idx, blk_idx, blk_type_idx, cx, cxy_size, 
		blk_type_blk_entries_size, i, j, from, h, v, t, tt, hh, vv ;
	T *ofdp, *otdp ;
	int *ip ;

	ot_idx = t_idx ;
	ofdp = fdp ;
	otdp = tdp ;
	while ( t_idx < to_tbl_size )
	{
		fdp = ofdp ;
		tdp = otdp ;

		blk_idx = t_idx / to_blk_entries_size ;
		t_idx -= blk_idx * to_blk_entries_size ; 	// index into this block

		blk_type_idx = get_blk_type_idx( blk_idx, blk_in_x, blk_in_y ) ;

		record_size = dcubep[ blk_type_idx ].md_v4_record_length ;

		blk_type_blk_entries_size = dcubep[ blk_type_idx ].md_v4_loopcnt *
			( record_size - NUM_OF_HVT_INDEX ) ;

		if ( t_idx < blk_type_blk_entries_size )
		{
#ifdef CUDA_DBG 
			// if (( blk_idx == 0 ) && ( t_idx < MD_DBG_SIZE ))
			if (( blk_idx == 1 ) && ( t_idx == 0 ))
			{
				i = 0 ;
				dbgp[ i++ ] = blk_type_blk_entries_size ;
				dbgp[ i++ ] = blk_idx ;
				dbgp[ i++ ] = t_idx ;
				dbgp[ i++ ] = record_size ;
				dbgp[ i++ ] = blk_type_idx ;
				dbgp[ i++ ] = to_blk_size ;
				dbgp[ i++ ] = from_blk_size ;
				dbgp[ i++ ] = to_blk_entries_size ;
			}
#endif 

			// tdp is pointing at the beginning of the to-block
			tdp += blk_idx * to_blk_size ;

			j = t_idx / ( record_size - NUM_OF_HVT_INDEX ) ;

			// t_idx ... the index of the element in this record
			t_idx -= ( j * ( record_size - NUM_OF_HVT_INDEX )) ;

			// tdp points to the record
			tdp += j * record_size ;

			ip = ( int * ) tdp ;
			t = ( *ip++ ) & CUBE_INFO_T_MSK ;
			v = *ip++ ;
			h = *ip ;

			// from block info
			cx = dcubep[ blk_type_idx ].x ;
			cxy_size = cx * dcubep[ blk_type_idx ].y ;

			// to block info
			mx = cx - md_x ;
			mxy_size = mx * ( dcubep[ blk_type_idx ].y - md_y ) ;

			fdp += blk_idx * from_blk_size ;	// adjust the from 
			// now ftp is pointing at the beginning of the from-block

			tdp += NUM_OF_HVT_INDEX ;

			tt = t_idx / mxy_size ; // which frame

			j = t_idx % mxy_size ;
			hh = j % mx ;	// which h
			vv = j / mx ;	// which v

			// from = t * cxy_size + ( v + vv ) * cx + h + hh ;
			// serial from = ( t + tt )  * mxy_size + ( v + vv ) * mx + h + hh ;
			from = ( t + tt )  * cxy_size + ( v + vv ) * cx + h + hh ;

			tdp[ t_idx ] = fdp[ from ] ; 
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}		
}

// step0: copy the data into the motion array ...
// block : the result of do_motion_idx <- edge-detection <- L-selection
// cube : the cube that is going to be moved by all h/v/t units
// fromp has the edged data ... top has the TVH index 
// md_x is csc.md_x * 2, same to md_y.  md_z is the same as csc.md_z
template<typename T>
int
h_do_motion_detection_step0_v4 ( T *fromp, T *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_cubep,	// cube in device	// will have the size of the 
	int from_block_size,
	int to_blk_entries_size, // exclude tvh ...
	int to_blk_size,	// include the tvh
   	int nblk_in_x, int nblk_in_y ) // new
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; //=  ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: fromp %p to %p size %d mdxyz %d %d %d blk_xy %d %d\n"
		"to_blk_entries_size %d to_blk_size %d from_block_size %d\n",
		__func__, fromp, top, tbl_size, md_x, md_y, md_z, nblk_in_x, nblk_in_y,
		to_blk_entries_size, to_blk_size, from_block_size ) ;
#endif 

#ifdef CUDA_DBG 
	if ( d_dbgp == NULL )
	{
		if (( nBlocks = cudaMalloc( &d_dbgp, MD_DBG_SIZE * sizeof( int ))) != cudaSuccess )
		{
			printf("%s: dbg cudaMalloc failed %d\n", __func__, nBlocks ) ;
			d_dbgp = NULL ;
		}
	}
	if ( d_dbgp != NULL )
		set_device_mem_i ( d_dbgp, MD_DBG_SIZE, 0 ) ;
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_motion_detection_step0_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top,
		tbl_size,	// does not include the 3 indexes 
		to_blk_entries_size, // do not include the 3 indexes
		to_blk_size,
		md_x, md_y, md_z,
		d_cubep, from_block_size,
	   	nblk_in_x, nblk_in_y
#ifdef CUDA_DBG 
		, d_dbgp
#endif 

	   	) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("motion_detect", d_dbgp, MD_DBG_SIZE ) ; 
#endif 
	return ( 1 ) ;
}

template int
h_do_motion_detection_step0_v4<int> ( int *fromp, int *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_cubep,	// cube in device	// will have the size of the 
	int from_block_size,
	int to_blk_entries_size, // exclude tvh ...
	int to_blk_size,	// include the tvh
   	int nblk_in_x, int nblk_in_y ) ; // new

template int
h_do_motion_detection_step0_v4<float> ( float *fromp, float *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_cubep,	// cube in device	// will have the size of the 
	int from_block_size,
	int to_blk_entries_size, // exclude tvh ...
	int to_blk_size,	// include the tvh
   	int nblk_in_x, int nblk_in_y ) ; // new

// 3 indexes + real data length == record_length
// loopcnt ... number of records in each blk 
// tbl_size ... number of records in blk_x * blk_y blks

__global__ void d_do_motion_idx_v4 ( int *dp, int tbl_size, int blk_size,
	int h_loop, int t_loop, int hv_size,
	int loopcnt,	// is the max num of loops in a block, regardless of corner/side/middle
	struct cube *cubep,
	int blk_in_x, int blk_in_y ) 
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ot_idx, blk_idx, blk, j ;
	int *odp ;
	int *ip ;

	ot_idx = t_idx ;
	odp = dp ;
	while ( t_idx < tbl_size )
	{
		blk_idx = t_idx / loopcnt ;	// which block
		dp = odp ;

		// j = ( t_idx % loopcnt ) % hv_size ;

		blk = get_blk_type_idx( blk_idx, blk_in_x, blk_in_y ) ;

		dp += blk_idx * blk_size ;

		t_idx -= blk_idx * loopcnt ;

		ip = ( int * ) dp + t_idx * cubep[blk].md_v4_record_length  ;

		if ( t_idx == ( cubep[blk].md_v4_loopcnt - 1 ))
		{
			*ip++ = CUBE_INFO_SET( blk ) ;	// tmporal

			*ip++ = (( hv_size / h_loop ) - 1 ) / 2 ;
			*ip++ = ( h_loop - 1 ) / 2 ;	 
		} else if ( t_idx < ( cubep[blk].md_v4_loopcnt - 1 ))
		{
			*ip++ = (( t_idx / hv_size ) + 1 ) | CUBE_INFO_SET( blk ) ;	// tmporal

			j = t_idx % hv_size ;
	
			*ip++ = j / h_loop ;	// vertical
			*ip++ = j % h_loop ;	// horizontal
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}  

// total size is the buffer size ... might not be the same as used.
// md_x/y/z aer each side
int
h_do_motion_idx_v4 ( int *dp, int total_size,
	int blk_in_x, int blk_in_y, struct cube *cubep,
	int md_x, int md_y, int md_z, struct cube *d_cubep )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int total_blk_size, record_length, k, j, i, nBlocks, loopcnt ;

	// the record length is the largest record amongst the inner/side/corner blks

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: total_size %d md x/y/z %d %d %d\n",
		__func__, total_size, md_x, md_y, md_z ) ;

	h_p_d_cube_config("device ", d_cubep ) ;
	h_p_h_cube_config("host ", cubep ) ;
#endif 

	total_blk_size = 0 ;
	record_length = 0 ;
	loopcnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		k = ( cubep[i].x - md_x * 2 ) *
			( cubep[i].y - md_y * 2 ) * ( cubep[i].md_v3_hv_cnt ) ; 

		if ( k > record_length )
			record_length = k ;

		j = ( k + NUM_OF_HVT_INDEX ) * cubep[i].md_v4_loopcnt ;

		if ( loopcnt < cubep[i].md_v4_loopcnt )
			loopcnt = cubep[i].md_v4_loopcnt ;

		if ( j > total_blk_size )
			total_blk_size = j ;
	}

	record_length += NUM_OF_HVT_INDEX ; // 3 indexes .. t/v/h in the beginning ...

	// the last record entry ( i.e. loopcnt -1 ) has a different format ...
	// ck the device code, when t_idx, ( loopcnt - 1)
	// loopcnt = ( md_x * 2 + 1 ) * ( md_y * 2 + 1 ) * cubep[0].md_v3_cnt + 1 ;

	i = total_blk_size * blk_in_x * blk_in_y ;

	if ( i > total_size )
	{
		fprintf( stderr, "%s: size needed %d got %d\n",
			__func__, i, total_size ) ;
		return ( 0 ) ;
	}

	i = loopcnt * blk_in_x * blk_in_y ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: loopcnt %d i %d rec %d md %d %d %d blk x/y %d %d total_blk_size %d",
		__func__, loopcnt, i, record_length, md_x, md_y, md_z, blk_in_x,
		blk_in_y, total_blk_size ) ;

	fprintf( stderr, " nBlocks %d \n", nBlocks ) ;
#endif 

	d_do_motion_idx_v4 <<< nBlocks, nThreadsPerBlock >>> (
		dp, i, total_blk_size,
		( md_x * 2 + 1 ), md_z, ( md_x * 2 + 1 ) * ( md_y * 2 + 1 ),
		loopcnt,
		d_cubep,
		blk_in_x, blk_in_y ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

// step one is to get y0-yk
// tbl_size: max entries size
template<typename T>
__global__ void d_do_l1_norm_step1_v4 ( T *o_dp, 
	int tbl_size,
	int to_blk_entries_size,	// max number of elements in a block ... some are not used ...
	int to_blk_size,	// block size ... including tvh and space not used
	struct cube *dcubep,
   	int blk_in_x, int blk_in_y )
{
	int o_t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int blk, blk_type_idx, record_length, t_idx, loopcnt, i ;
	T *dp, *op ;

	t_idx = o_t_idx ;
	while ( t_idx < tbl_size )
	{
		dp = o_dp ;

		blk = t_idx / to_blk_entries_size ;

		blk_type_idx = get_blk_type_idx ( blk, blk_in_x, blk_in_y ) ;

		t_idx -= ( blk * to_blk_entries_size ) ; // idx into this block

		record_length = dcubep[ blk_type_idx ].md_v4_record_length ;
		loopcnt = dcubep[ blk_type_idx ].md_v4_loopcnt ;

		i = ( loopcnt - 1 ) * ( record_length - NUM_OF_HVT_INDEX ) ; 

		if ( t_idx < i )
		{
			dp += blk * to_blk_size ; // dp points to this block

			dp += NUM_OF_HVT_INDEX ; // skip the TVH

			i = t_idx / ( record_length - NUM_OF_HVT_INDEX ) ;
			t_idx -= i * ( record_length - NUM_OF_HVT_INDEX ) ; // idx into this record ...

			op = dp + ( loopcnt - 1 ) * record_length ; // op points to orig

			dp += i * record_length ; // dp points to the beginning of the record

			dp += t_idx ;
			op += t_idx ;

			*dp -= *op ;

			if ( *dp < 0 )
				*dp = -*dp ;
		}
		o_t_idx += CUDA_MAX_THREADS ;
		t_idx = o_t_idx ;
	}   
}

// now make all entries positive for orig
// table_size is the all the entries of all orig in all ( nblk_in_x * nblk_in_y ) block

template<typename T>
__global__ void d_do_l1_norm_step1_1_v4 ( T *odp, 
	int tbl_size,
	int to_blk_entries_size,	// max number of elements in a record, across diff blk type
							// ... some are not used ...
	int to_blk_size,	// block size ... including tvh and space not used
	struct cube *dcubep,
   	int blk_in_x, int blk_in_y )
{
	int ot_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int record_length, blk_type_idx, loopcnt, t_idx, blk ;
	T *dp ;

	t_idx = ot_idx ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;
		t_idx = ot_idx ;

		blk = t_idx / to_blk_entries_size ; // to_blk_entries_size is the max of record_length of
			// all blk types

		blk_type_idx = get_blk_type_idx ( blk, blk_in_x, blk_in_y ) ;

		t_idx -= blk * to_blk_entries_size ;

		record_length = dcubep[ blk_type_idx ].md_v4_record_length ;
		if ( t_idx < ( record_length - NUM_OF_HVT_INDEX ))
		{	
			loopcnt = dcubep[ blk_type_idx ].md_v4_loopcnt ;

			dp += blk * to_blk_size ;
			dp += NUM_OF_HVT_INDEX ; // dp points to the block

			dp += record_length * ( loopcnt - 1 ) ; // dp points to the orig record, skip TVH

			dp += t_idx ;

			if ( *dp < 0 )
				*dp = -*dp ;	// save a step ... no need to abs()
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}

// total : entries_per_block * blk_in_x * blk_in_y
// entries_per_block : doex not include TVH
// blk_size: is the max block size
// max_record: the max record of all blk types 

template<typename T>
int
h_do_l1_norm_step1_v4( T *dp, int total, int entries_per_block, int blk_size,
	struct cube *d_cubep, int blk_in_x, int blk_in_y,
    int max_record ) 
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( total + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: total %d entries %d blk_size %d d_cubep %p blk x/y %d %d max %d\n", 
		__func__, total, entries_per_block, blk_size, d_cubep, blk_in_x,
		blk_in_y, max_record ) ;
#endif 

	if ( total != entries_per_block * blk_in_x * blk_in_y )
	{
		fprintf( stderr, "h_do_l1_norm_step1_v4: total err\n") ;
		return ( 0 ) ; 
	}

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step1_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, total, entries_per_block, blk_size, d_cubep, blk_in_x, blk_in_y ) ;

	cudaThreadSynchronize() ;

	total = max_record * blk_in_x * blk_in_y,

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step1_1_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, 
		total,
		max_record - NUM_OF_HVT_INDEX,
		blk_size,
		d_cubep,
		blk_in_x, blk_in_y ) ;

	return ( 1 ) ;

}

template int
h_do_l1_norm_step1_v4<int>( int *dp, int total, int entries_per_block, int blk_size,
	struct cube *d_cubep, int blk_in_x, int blk_in_y,
    int max_record )  ;

template int
h_do_l1_norm_step1_v4<float>( float *dp, int total, int entries_per_block, int blk_size,
	struct cube *d_cubep, int blk_in_x, int blk_in_y,
    int max_record )  ;


#ifdef CUDA_OBS 
template<typename T>
__global__ void ttt ( T *odp, , int *d_dbgp	)
{
		*d_dbgp = 222 ;
}
#endif 


__device__ int 
d_max_log2( int i )
{
	int k, j ;

	k = ( int )log2(( double ) i ) ;
	j = (int)pow(2.0, k ) ;

	if ( j == i )
		j >>=1 ;

	return ( j ) ;
}



int
set_up_cube_log( struct cube *tcubep, int hvt_adj )
{
	int j, i, cnt, start, max_cnt ;

	max_cnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		// y is cnt ... z is start ...

		cnt = tcubep[i].md_v4_record_length - hvt_adj ;

		start = max_log2( cnt ) ;
		if ( start != cnt )
			start = max_log2(( start / 2 ) - 1 ) ;
		else
			start >>= 1 ;
		
		tcubep[i].z = start ;	// where to start
		tcubep[i].y = cnt - start ;	// for how many

		j = tcubep[i].y * tcubep[i].md_v4_loopcnt ; // entries for this block

		if ( max_cnt < j )
			max_cnt = j ;

#ifdef CUDA_OBS 
		fprintf( stderr, "%s: i %d z %d y %d max %d cnt %d \n",
			__func__, i, tcubep[i].z, tcubep[i].y, max_cnt, cnt ) ;
#endif 
	}
	return ( max_cnt ) ;
}

int
set_up_cube_log_cont ( struct cube *tcubep )
{
	int j, i, max_cnt ;

	max_cnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		tcubep[i].z >>= 1 ;	// start
		tcubep[i].y = tcubep[i].z ;	// size

		j = tcubep[i].y * tcubep[i].md_v4_loopcnt ; // entries for this block

		if ( max_cnt < j )
			max_cnt = j ;

#ifdef CUDA_OBS 
	fprintf(stderr,"%s: i %d z %d y %d max %d \n",
		__func__, i, tcubep[i].z, tcubep[i].y, max_cnt ) ;
#endif 
	}
	return ( max_cnt ) ;
}

int 
find_thread_blk ( int threads )
{
	int k, j ;

	k = CUDA_MAX_THREADS_P_BLK * 2 ;

	if ( threads < k )
	{   
		j = max_log2 ( threads ) ;

		return ( j / 2 ) ;
	} else
		return ( CUDA_MAX_THREADS_P_BLK ) ;
}

// step 1.1 should be the abs() ... not needed, done in step 1 
// step 2 is to do the sum

template int
h_do_l1_norm_step2_v4_block_v2<int>( int *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template int
h_do_l1_norm_step2_v4_block_v2<float>( float *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template<typename T>
int
h_do_l1_norm_step2_v4_block_v1( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size )
{
	// cf h_grid_test@cuda_grid_blk_test.cu

	dim3 blk, grid ;
	int skip, todo, i, total_record, record_size ;

	blk.x = CUDA_MAX_THREADS_P_BLK ;
	blk.y = blk.z = 1 ;

	total_record = 0 ;
	record_size = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		if ( total_record < hcubep[i].md_v4_loopcnt )
			total_record = hcubep[i].md_v4_loopcnt ;

		if ( record_size < hcubep[i].md_v4_record_length )
			record_size = hcubep[i].md_v4_record_length ;
	}

	grid.x = total_record ;	// max number of records in block
	grid.y = ( record_size + 2 * CUDA_MAX_THREADS_P_BLK - 1 ) / ( 2 * CUDA_MAX_THREADS_P_BLK ) ;
		// number of cuda blocks per records
	grid.z = blk_in_x * blk_in_y ;	// how many blocks

	printf("%s : grid %d %d %d blk %d %d %d \n", __func__,
		grid.x,
		grid.y,
		grid.z,
		blk.x,
		blk.y,
		blk.z ) ;

	skip = 1 ;
	todo = record_size ;
	while ( todo > 1 )
	{
		d_do_l1_norm_step2_v4_block_v1<T><<<grid, blk>>> ( dp, d_cubep,
			blk_size, blk_in_x, blk_in_y, skip ) ;

		cudaThreadSynchronize() ;

		todo = ( todo + 2 * CUDA_MAX_THREADS_P_BLK - 1 ) / ( 2 * CUDA_MAX_THREADS_P_BLK ) ;
			// this much left to do

		skip *= ( 2 * CUDA_MAX_THREADS_P_BLK ) ;

		grid.y = ( todo + 2 * CUDA_MAX_THREADS_P_BLK - 1 ) / ( 2 * CUDA_MAX_THREADS_P_BLK ) ; // 

		printf("loop : skip %d todo %d \n", skip, todo ) ;
	}

	return ( 1 ) ;
}

template int
h_do_l1_norm_step2_v4_block_v1<int>( int *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template int
h_do_l1_norm_step2_v4_block_v1<float>( float *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template int
h_do_l1_norm_step2_v4_block<int>( int *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template int
h_do_l1_norm_step2_v4_block<float>( float *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;


template int
h_do_l1_norm_step2_v4<int>( int *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size, int hvt_adj ) ;

template int
h_do_l1_norm_step2_v4<float>( float *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size, int hvt_adj ) ;

#define MAX_L1_NORM			1000

// step 3 is to get 1-|y0-yk|/|y0| 
// odp points to real element, see caller

template<typename T>
__global__ void d_do_l1_norm_step3_v4 ( T *odp, int tbl_size, int entries_per_block,
	struct cube *d_cubep, int to_blk_size, int blk_in_x, int blk_in_y )
{
	int blk, ot_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *op, *dp ;
	int record_length, loopcnt, blk_type_idx, t_idx ;

	t_idx = ot_idx ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		blk = t_idx / entries_per_block ;	// which block

		blk_type_idx = get_blk_type_idx( blk, blk_in_x, blk_in_y ) ;

		loopcnt = d_cubep[ blk_type_idx ].md_v4_loopcnt ;
		record_length = d_cubep[ blk_type_idx ].md_v4_record_length ;

		t_idx -= blk * entries_per_block ;	// t_idx now index in this block

		if ( t_idx < ( loopcnt - 1 ))
		{
			dp += blk * to_blk_size ; // dp points to the start of block

			op = dp + (( loopcnt - 1 ) * record_length ) ;
				// op points to the original element 
			dp += t_idx * record_length ;	// dp points to the element
			
			// FIX ... if no int ... then there is no such problem

			// *dp = MAX_L1_NORM - ( MAX_L1_NORM * ( *dp )) / (*op) ;
			*dp = ((T)MAX_L1_NORM) - ( T )((( float )MAX_L1_NORM ) * ((( float )*dp ) / (( float ) *op ))) ;
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}

// entries_per_block : max of all types ... i.e. max ( loopcnt - 1 )
// total: entries_per_block * blk_x * blk_y
template<typename T>
int
h_do_l1_norm_step3_v4( T *dp, int total, int entries_per_block, 
	struct cube *d_cubep, int to_blk_size, int blk_in_x, int blk_in_y )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

#ifdef CUDA_OBS 
	printf("%s ::: total %d entries %d\n", __func__, total, entries_per_block ) ;
#endif 

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step3_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp + NUM_OF_HVT_INDEX, total, entries_per_block,
		d_cubep,
		to_blk_size,
		blk_in_x,
		blk_in_y ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

template int
h_do_l1_norm_step3_v4<int>( int *dp, int total, int entries_per_block, 
	struct cube *d_cubep, int to_block_size, int blk_in_x, int blk_in_y ) ;

template int
h_do_l1_norm_step3_v4<float>( float *dp, int total, int entries_per_block, 
	struct cube *d_cubep, int to_block_size, int blk_in_x, int blk_in_y ) ;

// move the max(t,v,h,value)+orig(t,v,h,value) of each block to the first block
template<typename T>
__global__ void d_do_l1_norm_step4_3_v4 ( T *odp, int total, 
	struct cube *d_cubep, int to_blk_size, int blk_in_x, int blk_in_y )
{
	int blk_type_idx, t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *tp, *dp ;
	int *ip, *itp ;

	while ( t_idx < total )
	{
		// t_idx is actually block_idx

		blk_type_idx = get_blk_type_idx ( t_idx, blk_in_x, blk_in_y ) ;

		dp = odp + t_idx * to_blk_size ; // dp points to the block

		tp = odp + t_idx * L1_NORM_STEP4_RETURN_ENTRY_SIZE * 2 ;	// first is the max, secondis no motion
		ip = ( int * )dp ;
		dp++ ;
		itp = ( int * )tp ;
		tp++ ;

		*itp = *ip & CUBE_INFO_T_MSK ;	// the max 
		*tp++ = *dp++ ; 
		*tp++ = *dp++ ; 
		*tp++ = *dp ; 

		// *dp points to the orig
		dp = odp + t_idx * to_blk_size + ( d_cubep[blk_type_idx].md_v4_loopcnt - 1 ) * 
			d_cubep[blk_type_idx].md_v4_record_length ;
		
		ip = ( int * )dp ;
		dp++ ;
		itp = ( int * )tp ;
		tp++ ;

		*itp = *ip & CUBE_INFO_T_MSK ;	// the max 
		*tp++ = *dp++ ; 
		*tp++ = *dp++ ; 
		*tp = *dp ; 

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// to find the max of each block block
// dp starts with idx 0 after NUM_OF_HVT_INDEX, see caller
// entries_per_block: is the max number of rows, in all blk type,
//		need to be processed in this block
// tbl_size: entries_per_block * blk_in_x * blk_in_y

template<typename T>
__global__ void d_do_l1_norm_step4_2_v4 ( T *odp, int tbl_size, int entries_per_block,
	struct cube *d_cubep, int to_blk_size, int blk_in_x, int blk_in_y )
{
	int t_idx, ot_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int blk, blk_type_idx ;
	T *tp, *dp ;

	t_idx = ot_idx ;
	while ( t_idx < tbl_size )
	{
		blk = t_idx / entries_per_block ;	// which blk

		dp = odp + blk * to_blk_size ; // dp points to block

		blk_type_idx = get_blk_type_idx ( blk, blk_in_x, blk_in_y ) ;

		t_idx -= blk * entries_per_block ; // idx to this blk

		if ( t_idx < d_cubep[ blk_type_idx ].y ) // size
		{
			tp = dp + d_cubep[ blk_type_idx ].md_v4_record_length * t_idx ; // points to the to element

			dp += d_cubep[ blk_type_idx ].md_v4_record_length * ( t_idx +
				d_cubep[ blk_type_idx ].z ) ;	// points to the from element

			if ( *tp < *dp )
			{
				*tp-- = *dp-- ;	// value
				*tp-- = *dp-- ;	// h
				*tp-- = *dp-- ;	// v
				*tp = *dp ;	// t
#ifdef CUDA_OBS 
				tip = ( int * )tp ;
				fip = ( int * )dp ;

				*tip-- = *fip-- ;	// value	// float or int QQQ ???
				*tip-- = *fip-- ;	// h
				*tip-- = *fip-- ;	// v
				*tip = *fip ;	// t
#endif 
			}
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}

// step 4.1: move the no_motion_row to the orig ... 
// total is blk_in_x * blk_in_y i.e. total is the number of blocks
// dp points to the correct data space behind NUM_OF_HVT_INDEX
// no_motion_idx is the block right after the orig in t-domain and no
//	shift in the h/v direction

template<typename T>
__global__ void d_do_l1_norm_step4_1_v4 ( T *odp, int total,
	int blk_in_x, int blk_in_y,
	int to_blk_size, struct cube *d_cubep, int no_motion_idx )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int record_size, loopcnt, i ;
	T *tdp, *dp ;

	while ( t_idx < total )
	{
		dp = odp ;

		i = get_blk_type_idx ( t_idx, blk_in_x, blk_in_y ) ; 	// which record in the block

		loopcnt = d_cubep[i].md_v4_loopcnt ; 
		record_size = d_cubep[i].md_v4_record_length ; 

		dp += t_idx * to_blk_size ;	// dp points to the block

		tdp = dp + ( loopcnt - 1 ) * record_size ; // tdp points to the orig record

		dp += no_motion_idx * record_size ; // dp points to the record

		*tdp++ = *dp++ ;	// t
		*tdp++ = *dp++ ;	// v
		*tdp++ = *dp++ ;	// h
		*tdp = *dp ;	// value

		t_idx += CUDA_MAX_THREADS ;
	}
}

// total is overall data area
// record_size does not include NUM_OF_HVT_INDEX
// orig: the block that every "moving" blocks compared with
template<typename T>
int
h_do_l1_norm_step4_v4( T *dp, int total,
	int *resp, int no_motion_idx,
	struct cube *d_cubep,
	int to_blk_size,
	int blk_in_x, int blk_in_y,
	struct cube *h_cubep
   	)
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int blocks, nBlocks, i, j, start, cnt ;
	struct cube t_cube[ CUBE_INFO_CNT ] ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: dp %p total %d rec %d orig %d hvt %d resp %p\n",
		__func__, total, total, record_size, orig, hvt_size, resp ) ;
#endif 

	blocks = blk_in_x * blk_in_y ;

#ifdef CUDA_OBS 
	h_p_d_cube_config ("device", d_cubep ) ;
	h_p_h_cube_config ("host", h_cubep ) ;
#endif 

	// step 4.1 ... move the no motion to the orig

	h_block_adj ( blocks, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step4_1_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, blocks,
		blk_in_x, blk_in_y,
		to_blk_size, 
		d_cubep, no_motion_idx ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	printf("%s : step 4.1 done \n", __func__ ) ;
#endif 

	// step 4.2 ... get the max 

	memcpy ( t_cube, h_cubep, sizeof ( *h_cubep ) * CUBE_INFO_CNT ) ;

	cnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		j = t_cube[i].md_v4_loopcnt - 1 ;

		start = max_log2( j ) ;
		if ( start != j  )
			start = max_log2(( start / 2 ) - 1 ) ;
		else
			start >>= 1 ;

		t_cube[i].z = start ;
		t_cube[i].y = j - start ; 	// cnt 

		if ( cnt < ( j - start ))
			cnt = j - start ;

#ifdef CUDA_OBS 
		printf("i %d cnt %d j %d start %d \n", i, cnt, j, start ) ;
#endif 
	}

	h_set_cube_config ( d_cubep, t_cube ) ;

	while ( cnt > 0 ) 
	{
#ifdef CUDA_OBS 
		printf("%s : cnt %d \n", __func__, cnt ) ;
#endif 

		// nBlocks= ( cnt * blocks + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

		j = cnt * blk_in_x * blk_in_y ;

		h_block_adj ( j , nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step4_2_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
			dp + NUM_OF_HVT_INDEX, j, cnt,
			d_cubep, to_blk_size, blk_in_x, blk_in_y ) ;

		cudaThreadSynchronize() ;

		cnt = 0 ;
		for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
		{
			t_cube[i].z >>= 1 ;
			t_cube[i].y = t_cube[i].z ; 	// cnt 

			if ( cnt < t_cube[i].y )
				cnt = t_cube[i].y ;
		}
		h_set_cube_config ( d_cubep, t_cube ) ;
	}

	h_set_cube_config ( d_cubep, h_cubep ) ;

#ifdef CUDA_OBS 
	printf("%s : step 4.2 done \n", __func__ ) ;
#endif 

	h_block_adj ( blocks, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step4_3_v4<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, blocks,
		d_cubep, to_blk_size, blk_in_x, blk_in_y ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	// 2: is the t/v/h/value for best one and the no move one
	if (( blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2 ) > (( record_size +
		NUM_OF_HVT_INDEX ) * hvt_size ))
	{
		fprintf(stderr, "%s: error: size mismatch %d %d\n", __func__, 
			( blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2 ),
			( record_size + NUM_OF_HVT_INDEX ) * hvt_size ) ;
		return ( 0 ) ;
	}
#endif 

#ifdef CUDA_OBS 
	printf("%s : step 4.3.1 done \n", __func__ ) ;

	printf("%s: outbuf %p device %p blks %d size %d\n", __func__,
		resp, dp, blocks, blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2 ) ;
#endif 

	if (( i = cudaMemcpy( resp, dp,
		blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2, 	
			// one for max, one for no motion
		cudaMemcpyDeviceToHost)) != cudaSuccess )
	{
		fprintf(stderr, "%s: memcpy failed %d\n", __func__, i ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_OBS 
	printf("%s : step 4.3 done \n", __func__ ) ;
#endif 

	return ( 1 ) ;
}

template int
h_do_l1_norm_step4_v4<int>( int *dp, int total,
	int *resp, int no_motion_idx,
	struct cube *d_cubep,
	int to_blk_size,
	int blk_in_x, int blk_in_y,
	struct cube *h_cubep ) ;

template int
h_do_l1_norm_step4_v4<float>( float *dp, int total,
	int *resp, int no_motion_idx,
	struct cube *d_cubep,
	int to_blk_size,
	int blk_in_x, int blk_in_y,
	struct cube *h_cubep ) ;

//

int
h_dbg_md_v4 ( const char *s, float *dp, struct cube *cubep, int blk_size, int blk_in_x, int blk_in_y,
	int pflag, int p1, int p2, int p3 )
{
	int *ip, *hp ;
	int total, blk_idx, i, j, k ;
	int t, v, h ;

	i = sizeof ( float ) * blk_size ;
	hp = ( int * ) malloc ( i ) ;

	if ( hp == NULL )
	{
		printf("%s ::: malloc failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	total = blk_in_x * blk_in_y ;

	printf("%s :: %s dp %p hp %p blk_size %d flag %x p1/2/3 %d %d %d\n",
		__func__, s, dp, hp, blk_size, pflag, p1, p2, p3 ) ;

	for ( i = 0 ; i < total ; i++ )
	{
		k = i / blk_in_x ;
		j = i % blk_in_x ;

		if (( k == 0 ) || ( k == ( blk_in_y - 1 )))
		{
			if (( j == 0 ) || ( j == ( blk_in_x - 1 )))
				blk_idx = CUBE_INFO_CORNER ;
			else
				blk_idx = CUBE_INFO_SIDE ;
		} else
		{
			if (( j == 0 ) || ( j == ( blk_in_x - 1 )))
				blk_idx = CUBE_INFO_SIDE ;
			else
				blk_idx = CUBE_INFO_INNER ;
		}

		if (( cubep[ blk_idx ].md_v4_loopcnt * cubep[ blk_idx ].md_v4_record_length ) >
			blk_size )
		{
			printf("%s :: cube err blk_idx %d loop %d rec %d blk_size %d \n",
				__func__,
				blk_idx, 
				cubep[ blk_idx ].md_v4_loopcnt,
			    cubep[ blk_idx ].md_v4_record_length,
			    blk_size ) ;

			free( hp ) ;

			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		printf("%s : dp %p ::: blk %d x %d y %d blk_idx %d \n",
			__func__, dp, i, k, j, blk_idx ) ;
#endif 

		dbg_get_d_data ( ( char *) dp, ( char * )hp, sizeof( float ) * blk_size ) ;

		ip = hp ;
		for ( k = 0 ; k < cubep[ blk_idx ].md_v4_loopcnt ; k++ )
		{
			t = *ip++ ;
			v = *ip++ ;
			h = *ip++ ;

			if ( pflag & P_TVH_IDX )
				printf("k %d ip %p:: t %x v %d h %d oip %p\n", k, ip - NUM_OF_HVT_INDEX, t, v, h, ip ) ;

			if (( pflag & P_BLK ) && ( i == p1 ) && (( k >= p2 ) && ( k <= p3 )))
			{
				printf("k %d ip %p:: t %x v %d h %d oip %p\n", k, ip - NUM_OF_HVT_INDEX, t, v, h, ip ) ;
				dbg_pdata_f ("record ... ", ( float *)ip, cubep[ blk_idx ].md_v4_record_length -
					NUM_OF_HVT_INDEX ) ;
			}

			if (( pflag & P_ROW ) && ( i == p1 ) && (( k >= p2 ) && ( k <= p3 )))
			{
				printf("k %d ip %p:: t %x v %d h %d oip %p === %f\n",
					k, ip - NUM_OF_HVT_INDEX, t, v, h, ip, *(( float * )ip)) ;
			}

			ip += cubep[ blk_idx ].md_v4_record_length - NUM_OF_HVT_INDEX ;
		}
		dp += blk_size ;
	}

	free( hp ) ;

	return ( 1 ) ;
}
