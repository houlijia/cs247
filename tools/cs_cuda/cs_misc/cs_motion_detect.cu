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
#include "cs_helper.h"
#include "cs_motion_detect.h"

// #define CUDA_DBG
// #define CUDA_DBG1

#define NUM_OF_HVT_INDEX	3

__global__ void d_do_motion_detection ( int *fdp, int *tdp, int tbl_size, 
	int record_size, // do include the 3 indexes
	int cx, int cxy_size,
	int mx, int mxy_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ot_idx, i, j, from, no_idx_size, *dp, h, v, t, tt, hh, vv ;

	ot_idx = t_idx ;
	while ( t_idx < tbl_size )
	{
		no_idx_size = record_size - NUM_OF_HVT_INDEX ;

		i = t_idx / no_idx_size ;
		dp = tdp + ( i * record_size ) ;
		t = *dp++ ;
		v = *dp++ ;
		h = *dp++ ;

		t_idx %= no_idx_size ;

		tt = t_idx / mxy_size ; // which block
		j = t_idx % mxy_size ;
		hh = j % mx ;	// which h
		vv = j / mx ;	// which v

		from = ( t + tt )  * cxy_size + ( v + vv ) * cx + h + hh ;

		dp[ t_idx ] = fdp[ from ] ; 
		// dp[ t_idx ] = from ; 

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}		
}

// do: 1 block at a time ... 
// block : the result of do_motion_idx <- edge-detection <- L-selection
// cube : the cube that is going to be moved by all h/v/t units
int
h_do_motion_detection ( int *fromp, int *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int record_size,	// includes the 3 indexes
	int blk_x, int blk_xy,	
	int cube_x, int cube_xy )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	fprintf(stderr, "%s: f %p t %p tblsize %d rec %d blk %d %d cube %d %d\n",
		__func__, fromp, top, tbl_size, record_size,
		blk_x, blk_xy, cube_x, cube_xy ) ;
#endif 

	if (( tbl_size % cube_xy ) || (( record_size - NUM_OF_HVT_INDEX ) % cube_xy ))
	{
		fprintf(stderr, "%s: error size %d cube %d rec %d\n", __func__,
			tbl_size, cube_xy, record_size ) ;
		return ( 0 ) ;
	}

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_motion_detection <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top, tbl_size, record_size, blk_x, blk_xy,
		cube_x, cube_xy ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("motion_detect", top, tbl_size ) ; 
#endif 
	return ( 1 ) ;
}

// 3 indexes + real data length == record_length

__global__ void d_do_motion_idx ( int *dp, int tbl_size, int record_length,
	int h_loop, int t_loop, int hv_size ) 
{
	int *odp, t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j ;

	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		dp += t_idx * record_length ;

		*dp++ = t_idx / hv_size ;	// tmporal

		j = t_idx % hv_size ;
	
		*dp++ = j / h_loop ;	// vertical
		*dp = j % h_loop ;	// horizontal

		t_idx += CUDA_MAX_THREADS ;
	}   
} 

int
h_do_motion_idx ( int *dp, int total_size, int record_length,
	int h_loop, int v_loop, int t_loop, int *orig_idx )
{
	int nThreadsPerBlock = 512;
	int nBlocks, loopcnt ;

	record_length += NUM_OF_HVT_INDEX ; // 3 indexes .. t/v/h in the beginning ...

	loopcnt = v_loop * h_loop * t_loop ;

	if (( record_length * loopcnt ) > total_size )
	{
		fprintf( stderr, "%s: size needed %d got %d\n",
			__func__, record_length * loopcnt, total_size ) ;
		return ( 0 ) ;
	}

	// nBlocks= ( loopcnt + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( loopcnt, nThreadsPerBlock, &nBlocks ) ;

	d_do_motion_idx <<< nBlocks, nThreadsPerBlock >>> (
		dp, loopcnt, record_length, h_loop, t_loop, h_loop * v_loop ) ;

	cudaThreadSynchronize() ;

	*orig_idx = ( v_loop / 2 ) * h_loop + ( h_loop / 2 ) ;

	return ( 1 ) ;
}

// step one is to get y0-yk

__global__ void d_do_l1_norm_step1 ( int *dp, int tbl_size, int record_length,
	int *op, int orig )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *odp, i, j ;

	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		j = t_idx / record_length ;

		if ( j != orig )
		{
			i = t_idx % record_length ;

			op += i ;
			dp = dp + j * ( record_length + NUM_OF_HVT_INDEX ) + 
				NUM_OF_HVT_INDEX + i ;

			*dp -= *op ;

			if ( *dp < 0 )
				*dp = -*dp ;	// save a step ... no need to abs()
		}

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// total and record_size does not have the NUM_OF_HVT_INDEX elements
void
h_do_l1_norm_step1( int *dp, int total, int record_size, int orig )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( total + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
	int *op ;

	op = dp + orig * ( record_size + NUM_OF_HVT_INDEX ) + NUM_OF_HVT_INDEX ;

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step1 <<< nBlocks, nThreadsPerBlock >>> (
		dp, total, record_size, op, orig ) ;

	cudaThreadSynchronize() ;
}

// step two is to get L1-norm(sum)
// all row, should be after the abs() is done
// tbl_size is the number of elements for this addition operation
// record_length includes the NUM_OF_HVT_INDEX
// dp starts with valid data, see caller

__global__ void d_do_l1_norm_step2 ( int *dp, int tbl_size, int record_length,
	int start, int cnt )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *odp, *tp, j ;

	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		j = t_idx / cnt ;

		tp = dp + j * record_length ;
		dp = tp + start ;

		j = t_idx % cnt ;

		tp[ j ] += dp [ j ] ;

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// step 1.1 should be to do the abs() 
// step 2 is to do the sum
// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
int
h_do_l1_norm_step2( int *dp, int total, int record_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks, i, start, row, cnt ;

	start = max_log2( record_size ) ;
	if ( start != record_size )
		start = max_log2(( start / 2 ) - 1 ) ;
	else
		start >>= 1 ;
	
	cnt = record_size - start ;
	row = total / record_size ;

	if ( total % record_size )
	{
		fprintf( stderr, "%s: error size %d %d \n", total, record_size ) ;
		return ( 0 ) ;
	}

	while ( cnt > 0 ) 
	{
		i = row * cnt ;

		printf("row %d cnt %d i %d start %d\n", row, cnt, i, start ) ;
		
		// nBlocks= ( i + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step2 <<< nBlocks, nThreadsPerBlock >>> (
			dp + NUM_OF_HVT_INDEX, i, record_size + NUM_OF_HVT_INDEX,
			start, cnt ) ;

		cudaThreadSynchronize() ;

		start >>= 1 ;
		cnt = start ;
	}
	return ( 1 ) ;
}

#define MAX_L1_NORM			1000

// step 3 is to get 1-|y0-yk|/|y0| 
// row_size is the number of rows ... 
// record_length includes the NUM_OF_HVT_INDEX
// dp starts with valid data, see caller

__global__ void d_do_l1_norm_step3 ( int *dp, int row_size, int record_length,
	int *op )
{
	int *odp, t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	odp = dp ;
	while ( t_idx < row_size )
	{
		dp = odp ;

		dp += t_idx * record_length ;

		// skip the orig
		if ( dp != op ) 
			*dp = MAX_L1_NORM - ( MAX_L1_NORM * ( *dp )) / (*op) ;

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
int
h_do_l1_norm_step3( int *dp, int total, int record_size, int orig )
{
	int nThreadsPerBlock = 512;
	int i, nBlocks ;

	if ( total % record_size )
	{
		fprintf( stderr, "%s: error size %d %d \n", total, record_size ) ;
		return ( 0 ) ;
	}

	i = total / record_size ;

	// nBlocks= ( i + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step3 <<< nBlocks, nThreadsPerBlock >>> (
		dp + NUM_OF_HVT_INDEX, i, record_size + NUM_OF_HVT_INDEX,
		dp + orig * ( record_size + NUM_OF_HVT_INDEX ) + NUM_OF_HVT_INDEX ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

// to find the min
// row_size is the number of rows for this addition operation
// record_length includes the NUM_OF_HVT_INDEX
// dp starts with idx 0 of NUM_OF_HVT_INDEX, see caller

__global__ void d_do_l1_norm_step4 ( int *dp, int cnt, int record_length,
	int start )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *odp, *tp ;

	odp = dp ;
	while ( t_idx < cnt )
	{
		dp = odp ;

		tp = dp + t_idx * record_length ;
		dp = tp + start * record_length ;

		if ( *dp >= 0 )
		{
			if ( *dp < *tp )
			{
				*tp-- = *dp-- ;	// value
				*tp-- = *dp-- ;	// h
				*tp-- = *dp-- ;	// v
				*tp-- = *dp ;	// t
			}
		}

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// step 4 is find the min 
// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
int
h_do_l1_norm_step4( int *dp, int total, int record_size, int orig,
	int *v )
{
	int nThreadsPerBlock = 512;
	int nBlocks, i, start, row, cnt ;
	int a[4] ;

	if ( total % record_size )
	{
		fprintf( stderr, "%s: error size %d %d \n", __func__, total,
			record_size ) ;
		return ( 0 ) ;
	}

	row = total / record_size ;

	if ( orig >= row )
	{
		fprintf( stderr, "%s: error orig %d row %d \n", __func__,
			orig, row ) ;
		return ( 0 ) ;
	}

	start = MAX_L1_NORM + 1 ;

	if (( i = cudaMemcpy( dp + orig * ( record_size + NUM_OF_HVT_INDEX ) +
		NUM_OF_HVT_INDEX, &start, sizeof( int ), cudaMemcpyHostToDevice))
		!= cudaSuccess )
	{
		printf("%s: download orig : %d\n", __func__, i ) ;
		return ( 0 ) ; 
	}

	start = max_log2( row ) ;
	if ( start != row )
		start = max_log2(( start / 2 ) - 1 ) ;
	else
		start >>= 1 ;
	
	cnt = row - start ;

	while ( cnt > 0 ) 
	{
		printf("row %d cnt %d start %d\n", row, cnt, start ) ;
		
		// nBlocks= ( cnt + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

		h_block_adj ( cnt, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step4 <<< nBlocks, nThreadsPerBlock >>> (
			dp + NUM_OF_HVT_INDEX, cnt, record_size + NUM_OF_HVT_INDEX,
			start ) ;

		cudaThreadSynchronize() ;

		start >>= 1 ;
		cnt = start ;
	}

	if (( i = cudaMemcpy( a, dp, 4 * sizeof( int ), cudaMemcpyDeviceToHost ))
		!= cudaSuccess )
	{
		printf("%s: upload orig : %d\n", __func__, i ) ;
		return ( 0 ) ; 
	}

	*v++ = a[0] ;
	*v++ = a[1] ;
	*v++ = a[2] ;
	*v++ = a[3] ;

	return ( 1 ) ;
}
