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
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_motion_detect_v3.h"
#include "cs_analysis.h"

/*
Fri Apr 24 17:46:50 EDT 2015

version 2 will only match the first md block with the subsequent md_z -1 block to do
the L1 norm calculation.  which works fine, but not too accurate, v3 will change that
to compare the "center" md blocks across the termperal domain
*/

// #define CUDA_DBG1
// #define CUDA_DBG
// #define CUDA_OBS

// md_x/y/z: total size ... so it is md_x'*2, md_y'*2 and md_z
// tbl_size,	// does not include the 3 indexes 
// record_size, // do not include the 3 indexes
// hvt_size,	// number of combination of h/v/t
template<typename T>
__global__ void d_do_motion_detection_step0_v3 (
	T *ofdp, T *otdp, 
	int tbl_size,
	int cx, int cxy, // orig edged block x/y sizes
	int mx, int mxy, // new motion detected block x/y sizes
	int record_size, // = mxy * md_v3_cnt ;
	int md_v3_cnt )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int rec_idx, ot_idx, h, v, t ;
	T *fdp, *tdp ;
	int *ip ;

	ot_idx = t_idx ;
	while ( t_idx < tbl_size )
	{
		fdp = ofdp ;
		tdp = otdp ;

		rec_idx = t_idx / record_size ;
		t_idx = t_idx % record_size ; 	// index into this record

		// tdp is pointing at the beginning of the to-block
		tdp += ( record_size + NUM_OF_HVT_INDEX ) * rec_idx ; 

		ip = ( int * )tdp ;

		t = *ip++ ;
		v = *ip++ ;
		h = *ip ;

		tdp += ( t_idx + NUM_OF_HVT_INDEX ) ;	// this is the destination

		t += ( t_idx / mxy ) ;
		
		t_idx %= mxy ;

		v += ( t_idx / mx ) ;
		h += ( t_idx % mx ) ;

		fdp += t * cxy ;	// adjust the from 

		fdp += ( v * cx + h ) ; // now tfp is pointing at the src in the from-block

		*tdp = *fdp ;

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}		
}

template<typename T>
int
h_do_motion_detection_step0_v3 ( T *fromp, T *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int edge_x, int edge_xy,	// orig edged block x/y size
	int md_x, int md_xy,	// new md block x/y size
	int md_v3_cnt )	// max number of in_x*in_y in a record
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; //=  ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: fromp %p to %p size %d edge %d %d md %d %d v3_cnt %d \n",
		__func__, fromp, top, tbl_size, edge_x, edge_xy, md_x, md_xy, md_v3_cnt ) ;

#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_motion_detection_step0_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top,
		tbl_size,	// does not include the 3 indexes 
		edge_x, edge_xy, 
		md_x, md_xy, 
		md_xy * md_v3_cnt,
		md_v3_cnt ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("motion_detect", top, tbl_size ) ; 
#endif 
	return ( 1 ) ;
}

template int
h_do_motion_detection_step0_v3<float> ( float *fromp, float *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int edge_x, int edge_xy,	// orig edged block x/y size
	int md_x, int md_xy,	// new md block x/y size
	int md_v3_cnt ) ;	// max number of in_x*in_y in a record

template int
h_do_motion_detection_step0_v3<int> ( int *fromp, int *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int edge_x, int edge_xy,	// orig edged block x/y size
	int md_x, int md_xy,	// new md block x/y size
	int md_v3_cnt ) ;	// max number of in_x*in_y in a record

// 3 indexes + real data length == record_length
// loopcnt ... number of records in T-specific block
// tbl_size ... number of records in the whole table
__global__ void d_do_motion_idx_v3 ( int *dp, int tbl_size, int loopcnt,
	int record_length, int h_loop, int max_loopcnt )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ot_idx, blk_idx ;
	int *odp ;

	ot_idx = t_idx ;
	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		dp += t_idx * record_length ;

		if ( t_idx >= max_loopcnt )
		{
			*dp++ = 0 ;
			*dp++ = (( loopcnt / h_loop ) - 1 ) / 2 ;
			*dp++ = ( h_loop - 1 ) / 2 ;	 
		} else 
		{
			// the index is in T, V and H order

			blk_idx = t_idx / loopcnt ;	// which T-specific block
			t_idx -= blk_idx * loopcnt ;

			*dp++ = blk_idx + 1 ;

			*dp++ = t_idx / h_loop ;	// vertical
			*dp++ = t_idx % h_loop ;	// horizontal
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}  

// total size is the buffer size ... might not be the same as used.
// md_x/y/z are each side
// cube_idx ... which sice/corner/inner cube is this? ... we can only handle
// one cube at a time for v3
int
h_do_motion_idx_v3 ( int *dp, int total_size,
	int *orig_idx, struct cube *cubep,
	int md_x, int md_y, int md_z, int *record_sizep, int cube_idx )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int max_loopcnt, record_length, k, i, nBlocks, loopcnt ;

	// the record length is the largest record amongst the inner/side/corner blks

	fprintf( stderr, "%s: total_size %d md x/y/z %d %d %d\n",
		__func__, total_size, md_x, md_y, md_z ) ;

	h_p_h_cube_config ("h_do_motion_idx_v3", cubep ) ;

	record_length = ( cubep[ cube_idx ].x - md_x * 2 ) *
		( cubep[ cube_idx ].y - md_y * 2 ) * ( cubep[ cube_idx ].md_v3_hv_cnt ) ; 

	*record_sizep = record_length ;

	record_length += NUM_OF_HVT_INDEX ; // 3 indexes .. t/v/h in the beginning ...

	// the last md_v3_cnt record entries have a different format ...
	// ck the device code.

	loopcnt = ( md_x * 2 + 1 ) * ( md_y * 2 + 1 ) ;

	k = cubep[ cube_idx ].md_v3_cnt ;

	i = record_length * ( loopcnt * k + 1 ) ;	// 1 is for orig
		// + md_v3_cnt, since each loopcnt will need one
		// orig ... for calculate the sum

	max_loopcnt = k * loopcnt ;

	if ( i > total_size )
	{
		fprintf( stderr, "%s: size needed %d got %d\n",
			__func__, i, total_size ) ;
		return ( 0 ) ;
	}

	i /= record_length ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	fprintf( stderr, "%s: loopcnt %d i %d rec %d md %d %d %d maxloop %d k %d\n",
		__func__, loopcnt, i, record_length, md_x, md_y, md_z, max_loopcnt,k ) ;
 
	d_do_motion_idx_v3 <<< nBlocks, nThreadsPerBlock >>> (
		dp, i, loopcnt, record_length, ( md_x * 2 + 1 ), max_loopcnt ) ;

	cudaThreadSynchronize() ;

	*orig_idx = max_loopcnt ;

	return ( 1 ) ;
}

// step one is to get y0-yk
// no need to worry about the different cube size ... inner/side/corner
// it will be all junk anyway ...
// tbl_size does not include orig
// rec_size does not have the T/V/H
template<typename T>
__global__ void d_do_l1_norm_step1_v3 ( T *odp, int tbl_size, int record_length,
	int orig )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j ;
	T *dp, *op ;

	while ( t_idx < tbl_size )
	{
		dp = odp ;

		i = t_idx / record_length ;
		dp += i * ( record_length + NUM_OF_HVT_INDEX ) ;

		j = t_idx % record_length ;	// index into the current record

		dp += NUM_OF_HVT_INDEX + j;

		op = odp + orig * ( record_length + NUM_OF_HVT_INDEX ) ;	// k is 1 relative
		op += NUM_OF_HVT_INDEX + j ;

		*dp -= *op ;

		if ( *dp < 0 )
			*dp = -*dp ;	// save a step ... no need to abs()

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// now make all entries positive for orig
// table_size is the all the entries in orig
// odp does not have NUM_OF_HVT_INDEX, so ready to go

template<typename T>
__global__ void d_do_l1_norm_step1_1_v3 ( T *odp, int tbl_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *dp ;

	while ( t_idx < tbl_size )
	{
		dp = odp + t_idx ;

		if ( *dp < 0 )
			*dp = -*dp ;	// save a step ... no need to abs()

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// total and record_size does not have the NUM_OF_HVT_INDEX elements
// total does not have the orig * record_size

template<typename T>
int
h_do_l1_norm_step1_v3( T *dp, int total, int record_size, int orig )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( total + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	if ( total % record_size )
	{
		fprintf( stderr, "%s: error total %d rec %d \n",
			__func__, total, record_size ) ;
		return ( 0 ) ; 
	}

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step1_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, total, record_size, orig ) ;

	cudaThreadSynchronize() ;

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	dp += orig * ( record_size + NUM_OF_HVT_INDEX ) ;
	dp += NUM_OF_HVT_INDEX ;

	d_do_l1_norm_step1_1_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, record_size ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;

}

template int
h_do_l1_norm_step1_v3<int>( int *dp, int total, int record_size, int orig ) ;

template int
h_do_l1_norm_step1_v3<float>( float *dp, int total, int record_size, int orig ) ;

// step two is to get L1-norm(sum)
// all row, should be after the abs() is done
// tbl_size is the number of elements for this addition operation
// record_length includes the NUM_OF_HVT_INDEX
// cnt is the max_cnt for each record, regardless inner/side/corner

template<typename T>
__global__ void d_do_l1_norm_step2_v3 ( T *odp, int tbl_size, int record_length,
	int cnt, int start ) 
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j ;
	T *fp, *dp ;

	while ( t_idx < tbl_size )
	{
		dp = odp ;

		j = t_idx / cnt ;

		dp += record_length * j ;

		j = t_idx % cnt ;

		dp += NUM_OF_HVT_INDEX ;
		fp = dp + start ;

		dp[ j ] += fp [ j ] ;

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// add up the sum in step 2 ...
// step 1.1 should be the abs() ... not needed, done in step 1 
// step 2 is to do the sum
// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
// total = record_size * row
template<typename T>
int
h_do_l1_norm_step2_v3( T *dp, int record_size, int row )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks, i, start ;
	int max_cnt ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: dp %p total %d record %d \n",
		__func__, dp, row, record_size ) ;
#endif 

	start = max_log2( record_size ) ;

	if ( start != record_size )
		start = max_log2(( start / 2 ) - 1 ) ;
	else
		start >>= 1 ;

	max_cnt = record_size - start ;

	while ( max_cnt > 0 ) 
	{
		i = row * max_cnt ;

#ifdef CUDA_DBG1 
		fprintf( stderr, "row %d cnt %d i %d \n", row, max_cnt, i ) ;
#endif 
		
		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step2_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
			dp, i, record_size + NUM_OF_HVT_INDEX, max_cnt, start ) ;

		cudaThreadSynchronize() ;

		max_cnt = 0 ;

		start >>= 1 ;
		max_cnt = start ;
	}

	return ( 1 ) ;
}

template int
h_do_l1_norm_step2_v3<int>( int *dp, int record_size, int row ) ;

template int
h_do_l1_norm_step2_v3<float>( float *dp, int record_size, int row ) ;

#define MAX_L1_NORM			1000

// step 3 is to get 1-|y0-yk|/|y0| 
// row_size is the number of rows ... not includes orig
// record_length includes the NUM_OF_HVT_INDEX
// dp starts with valid data, see caller

template<typename T>
__global__ void d_do_l1_norm_step3_v3 ( T *odp, int row_size, int record_length,
	int orig )
{
	int ot_idx, t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *op, *dp ;

	ot_idx = t_idx ;
	while ( t_idx < row_size )
	{
		op = odp + orig * record_length ;
		dp = odp + t_idx * record_length ;
			
		// when T is int ... we need this ...

		*dp = ((T)MAX_L1_NORM) - ( T )((( float )MAX_L1_NORM ) * ((( float )*dp ) / (( float ) *op ))) ;

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}

// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
// row_cnt does not include the orig ...

template<typename T>
int
h_do_l1_norm_step3_v3( T *dp, int record_size, int orig, int row_cnt )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;

#ifdef CUDA_DBG 
	printf("%s : dp %p rec_size %d orig %d row_cnt %d \n",
		__func__, dp, record_size, orig, row_cnt ) ;
#endif 

	h_block_adj ( row_cnt, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step3_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp + NUM_OF_HVT_INDEX, row_cnt, record_size + NUM_OF_HVT_INDEX,
		orig ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

template int
h_do_l1_norm_step3_v3<int>( int *dp, int record_size, int orig, int row ) ;

template int
h_do_l1_norm_step3_v3<float>( float *dp, int record_size, int orig, int row ) ;

// record_length includes the NUM_OF_HVT_INDEX
// dp starts before the NUM_OF_HVT_INDEX,  see caller
// total is the number of hvt_blocks

template<typename T>
__global__ void d_do_l1_norm_step4_3_v3 ( T *odp, int total, int record_length,
	int orig )
{
	int *fip, *tip, t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *fp ;

	while ( t_idx < total )
	{
		// depends on the size of int the size of float ...

		fp = odp + L1_NORM_STEP4_RETURN_ENTRY_SIZE ;

		tip = ( int *)fp ;

		fp = odp + orig * record_length ;

		fip = ( int *)fp ;	// no move

		*tip++ = *fip++ ;
		*tip++ = *fip++ ; 
		*tip++ = *fip++ ; 
		*tip++ = *fip ; 

		t_idx += CUDA_MAX_THREADS ;
	}   
}

template<typename T> int
h_do_l1_norm_step4_3_v3 ( T *odp, int record_length, int orig )
{
	int i ;
	T *tdp, *dp ;

	tdp = odp + orig * ( record_length + NUM_OF_HVT_INDEX ) ;
	dp = odp + L1_NORM_STEP4_RETURN_ENTRY_SIZE ;

	if (( i = cudaMemcpy(( void *)dp, ( void *)tdp,
		L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ), 	
		cudaMemcpyDeviceToDevice)) != cudaSuccess )
	{
		printf("%s: error cudaMemcpy %d \n", __func__, i ) ;
		return ( 0 ) ;
	} else
		return ( 1 ) ;
}
template int
h_do_l1_norm_step4_3_v3<int> ( int *odp, int record_length, int orig ) ;

template int
h_do_l1_norm_step4_3_v3<float> ( float *odp, int record_length, int orig ) ;

// to find the max of each hvt_size block
// record_length does include the NUM_OF_HVT_INDEX
// dp starts with after TVH header, see caller
// cnt is the number of rows need to be processed in this row_cnt rows
// total is the number of entries in ( hvt_size * cnt ) need to be 
//		processed at this run

template<typename T>
__global__ void d_do_l1_norm_step4_2_v3 ( T *odp, int record_length,
	int start, int cnt )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *tip, *fip ;
	T *tp, *dp ;
	int total = cnt ;

	while ( t_idx < total )
	{
		tp = odp + t_idx * record_length ;	// destination
		dp = tp + start * record_length ;   // src

		if ( *tp < *dp )
		{
			tip = ( int * )tp ;
			fip = ( int * )dp ;

			*tip-- = *fip-- ;	// value	// float or int QQQ ???
			*tip-- = *fip-- ;	// h
			*tip-- = *fip-- ;	// v
			*tip = *fip ;	// t
		}

		t_idx += CUDA_MAX_THREADS ;
	}   
}

template<typename T>
__global__ void d_do_l1_norm_step4_2_v3_1 ( T *odp, int record_length,
	int str, int total )
{
	// int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int cnt, size, start, start_idx, k, *tip, *fip ;
	T *tp, *dp ;

	start_idx = blockIdx.x * blockDim.x * 2 ;

	k = total - start_idx ;

	size = ( k > blockDim.x * 2 ) ? blockDim.x * 2 : k ;	// size is the number of entries
		// that the threads in this block have to handle

	while ( size > 1 )
	{
		start = ( size + 1 ) / 2 ;

		cnt = size - start ;

		if ( threadIdx.x < cnt )
		{
			tp = odp + ( start_idx + threadIdx.x ) * str * record_length ;	// destination
			dp = tp + start * str * record_length ;   // src

			if ( *tp < *dp )
			{
				tip = ( int * )tp ;
				fip = ( int * )dp ;

				*tip-- = *fip-- ;	// value	// float or int QQQ ???
				*tip-- = *fip-- ;	// h
				*tip-- = *fip-- ;	// v
				*tip = *fip ;	// t
			}
		}

		size -= cnt ;

		__syncthreads() ; // wait for all threads in this block

		// t_idx += CUDA_MAX_THREADS ;
	}   
}

// step 4.1: move the no_motion_row to the orig ... 
// record_length does not have the NUM_OF_HVT_INDEX
// dp points to the TVH header
// no_motion_idx is the block right after the orig in t-domain and no
//	shift in the h/v direction

template<typename T>
__global__ void d_do_l1_norm_step4_1_v3 ( T *odp, int record_length,
	int orig, int no_motion_idx )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *tip, *fip ;
	T *tdp, *dp ;

	while ( t_idx < 1 )
	{
		tdp = odp + orig * ( record_length + NUM_OF_HVT_INDEX ) ;
		dp = odp + no_motion_idx * ( record_length + NUM_OF_HVT_INDEX ) ;
	
		tip = ( int *)tdp ;
		fip = ( int *)dp ;
	
		*tip++ = *fip++ ; 	// t
		*tip++ = *fip++ ; 	// v
		*tip++ = *fip++ ; 	// h
		*tip = *fip ; 	// value, depending on the size of int to be the same as float ... not good.

		t_idx += CUDA_MAX_THREADS ;
	}
}

int
h_do_l1_norm_step4_1_v3_1 ( float *odp, int record_length, int no_motion_idx, float *resp )
{
	int i ;
	float *dp ;

	resp += L1_NORM_STEP4_RETURN_ENTRY_SIZE ;

	dp = odp + no_motion_idx * ( record_length + NUM_OF_HVT_INDEX ) ;

	if (( i = cudaMemcpy(( void *)resp, ( void *)dp,
		L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ), 	
		cudaMemcpyDeviceToHost)) != cudaSuccess )
	{
		printf("%s: error cudaMemcpy %d \n", __func__, i ) ;
		return ( 0 ) ;
	} else
		return ( 1 ) ;	
}

// copy no motion to orig 
template<typename T> int
h_do_l1_norm_step4_1_v3 ( T *odp, int record_length, int orig, int no_motion_idx )
{
	int i ;
	T *tdp, *dp ;

	tdp = odp + orig * ( record_length + NUM_OF_HVT_INDEX ) ;
	dp = odp + no_motion_idx * ( record_length + NUM_OF_HVT_INDEX ) ;

	if (( i = cudaMemcpy(( void *)tdp, ( void *)dp,
		L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ), 	
		cudaMemcpyDeviceToDevice)) != cudaSuccess )
	{
		printf("%s: error cudaMemcpy %d \n", __func__, i ) ;
		return ( 0 ) ;
	} else
		return ( 1 ) ;	
}

template int
h_do_l1_norm_step4_1_v3<int> ( int *odp, int record_length, int orig, int no_motion_idx ) ;

template int
h_do_l1_norm_step4_1_v3<float> ( float *odp, int record_length, int orig, int no_motion_idx ) ;

// total is overall data area
// record_size does not include NUM_OF_HVT_INDEX
// row does not include orig
// total is overall data area
// record_size does not include NUM_OF_HVT_INDEX
// row does not include orig
// orig: the block that every "moving" blocks compared with
template<typename T>
int
h_do_l1_norm_step4_v3( T *dp, int record_size, int orig,
	int row_cnt, int *resp, int no_motion_idx, int timer_idx )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int skip_size, nBlocks, i, cnt ;

#ifdef CUDA_OBS 
	int start ;
#endif 

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: dp %p rec  %d orig %d row_cnt %d resp %p no_motion %d\n",
		__func__, dp, record_size, orig, row_cnt, resp, no_motion_idx ) ;
#endif 

	// step 4.1 ... 0 out all orig and negative entries ...

#ifdef CUDA_OBS 
	cs_p_d_tvh ("before orig", ( int *)dp + orig * ( record_size + NUM_OF_HVT_INDEX ),
		record_size, 1, 6 ) ;

	cs_p_d_tvh ("no_motion", ( int *)dp + no_motion_idx * ( record_size + NUM_OF_HVT_INDEX ),
		record_size, 1, 6 ) ;
#endif 

	omp_timer_on ( timer_idx ) ;

#ifdef CUDA_OBS 
	h_block_adj ( 1, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step4_1_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, record_size, orig, no_motion_idx ) ;

	cudaThreadSynchronize() ;
#endif 

	// h_do_l1_norm_step4_1_v3<float> ((float*) dp, record_size, orig, no_motion_idx ) ;
	h_do_l1_norm_step4_1_v3_1 ((float*) dp, record_size, no_motion_idx, (float *)resp ) ;

	omp_timer_off ( timer_idx ) ;

#ifdef CUDA_OBS 
	cs_p_d_tvh ("after orig", ( int *)dp + orig * ( record_size + NUM_OF_HVT_INDEX ),
		record_size, 1, 6 ) ;

	cs_p_d_tvh ("no_motion", ( int *)dp + no_motion_idx * ( record_size + NUM_OF_HVT_INDEX ),
		record_size, 1, 6 ) ;
#endif 

	printf("%s : step 4.1 done \n", __func__ ) ;

	// step 4.2 ... get the max 

	omp_timer_on ( timer_idx+1 ) ;

#ifdef CUDA_OBS 
	start = max_log2( row_cnt ) ;
	if ( start != row_cnt )
		start = max_log2(( start / 2 ) - 1 ) ;
	else
		start >>= 1 ;
	
	cnt = row_cnt - start ;

	while ( cnt > 0 ) 
	{
#ifdef CUDA_DBG 
		printf("%s : cnt %d start %d\n", __func__, cnt, start ) ;
#endif 

		h_block_adj ( cnt, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step4_2_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
			dp + NUM_OF_HVT_INDEX, record_size + NUM_OF_HVT_INDEX,
			start, cnt ) ;

		cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
		cs_p_d_tvh ("SORTING", ( int *)dp, record_size, start + cnt, 0 ) ;
#endif 
		start >>= 1 ;
		cnt = start ;
	}
#endif 

	skip_size = nThreadsPerBlock * 2 ;
	cnt = row_cnt ;
	i = 1 ;

	while ( cnt != 1 )
	{
		h_block_adj (( cnt + 1 )/2, nThreadsPerBlock, &nBlocks ) ;

		printf("%s :: cnt %d stride %d, nblk %d \n", __func__,
			cnt, i, nBlocks ) ;

		d_do_l1_norm_step4_2_v3_1<T> <<< nBlocks, nThreadsPerBlock >>> 
			(dp + NUM_OF_HVT_INDEX, record_size + NUM_OF_HVT_INDEX, i, cnt ) ;
		
		cudaThreadSynchronize() ;

		i *= skip_size ;
	   	
		cnt = ( cnt + skip_size - 1 ) / skip_size ;	

		printf("%s :: last cnt %d stride %d, nblk %d \n", __func__,
			cnt, i, nBlocks ) ;
	}

	omp_timer_off ( timer_idx+1 ) ;

#ifdef CUDA_DBG 
	cs_p_d_tvh ("after MAX", ( int *)dp, record_size, 10, 6 ) ;
#endif 

	printf("%s : step 4.2 done \n", __func__ ) ;

	omp_timer_on ( timer_idx+2 ) ;

#ifdef CUDA_OBS 

	h_block_adj ( 1, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step4_3_v3<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, 1, record_size + NUM_OF_HVT_INDEX, orig ) ;

	cudaThreadSynchronize() ;
#endif 

	// h_do_l1_norm_step4_3_v3<float> (( float *)dp, record_size, orig ) ;

	omp_timer_off ( timer_idx+2 ) ;

	printf("%s : step 4.3.1 done \n", __func__ ) ;
	printf("%s: outbuf %p device %p\n", __func__, resp, dp ) ;

	omp_timer_on ( timer_idx+3 ) ;

	// one for max, one for no motion
	// just the max, the motion is already in place
	if (( i = cudaMemcpy( resp, dp,
		// L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2, 	
		L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ), 	
		cudaMemcpyDeviceToHost)) != cudaSuccess )
	{
		fprintf(stderr, "%s: memcpy failed %d\n", __func__, i ) ;
		return ( 0 ) ;
	}

	omp_timer_off ( timer_idx+3 ) ;

#ifdef CUDA_DBG 
	printf("IDX resp %p t %d v %d h %d ", resp, resp[0], resp[1], resp[2] ) ;
	printf("v %f\n", ( float )resp[3] ) ;
	printf("IDX t %d v %d h %d ", resp[4], resp[5], resp[6] ) ;
	printf("v %f\n", ( float )resp[7] ) ;
#endif 

	printf("%s : step 4.3 done \n", __func__ ) ;

	return ( 1 ) ;
}

template int
h_do_l1_norm_step4_v3<int>( int *dp, int record_size, int orig,
	int row_cnt, int *resp, int no_motion_idx, int timer_idx ) ;

template int
h_do_l1_norm_step4_v3<float>( float *dp, int record_size, int orig,
	int row_cnt, int *resp, int no_motion_idx, int timer_idx ) ;
