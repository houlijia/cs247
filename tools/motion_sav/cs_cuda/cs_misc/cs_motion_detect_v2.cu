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
#include "cs_motion_detect_v2.h"
#include "cs_analysis.h"

/*
Fri Apr 24 17:46:50 EDT 2015

version 2 will only match the first md block with the subsequent md_z -1 block to do
the L1 norm calculation.  which works fine, but not too accurate, v3 will change that
to compare the "center" md blocks across the termperal domain
*/

#define CUDA_DBG
// #define CUDA_OBS

// md_x/y/z: total size ... so it is md_x'*2, md_y'*2 and md_z
// tbl_size,	// does not include the 3 indexes 
// record_size, // do not include the 3 indexes
// hvt_size,	// number of combination of h/v/t
// from_blk_size,	// the size of the input inner block ... after edge
template<typename T>
__global__ void d_do_motion_detection_step0_v2 (
	T *fdp, T *tdp, 
	int tbl_size,
	int record_size,
	int hvt_size,
	int md_x, int md_y, int md_z,
	struct cube *dxyzp,
	int from_blk_size )
{
	int mx, mxy_size ;
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ot_idx, blk_idx, blk_size, blk_type_idx, cx, cxy_size, 
		md_z_len, i, j, from, h, v, t, tt, hh, vv ;
	T *ofdp, *otdp ;
	int *ip ;
#ifdef CUDA_OBS 
	int *dbp ;
#endif 

#ifdef CUDA_OBS 
	dbp = fdp ;
#endif 

	ot_idx = t_idx ;
	ofdp = fdp ;
	otdp = tdp ;
	while ( t_idx < tbl_size )
	{
		fdp = ofdp ;
		tdp = otdp ;

		blk_size = hvt_size * record_size ;	// LDL can be moved out of while loop

		blk_idx = t_idx / blk_size ;
		t_idx -= blk_idx * blk_size ; 	// index into this block

		// tdp is pointing at the beginning of the to-block
		tdp += blk_idx * ( record_size + NUM_OF_HVT_INDEX ) * hvt_size ; 

		ip = ( int * ) tdp ;
		j = *ip ;
		blk_type_idx = CUBE_INFO_GET( j ) ;
 
		// from block info
		cx = dxyzp[ blk_type_idx ].x ;
		cxy_size = cx * dxyzp[ blk_type_idx ].y ;

		// to block info
		mx = cx - md_x ;
		mxy_size = mx * ( dxyzp[ blk_type_idx ].y - md_y ) ;
		md_z_len = dxyzp[ blk_type_idx ].z - md_z + 1 ;

#ifdef CUDA_OBS 
		*dbp++ = blk_type_idx ;
		*dbp++ = cx ;
		*dbp++ = cxy_size ;
		*dbp++ = blk_idx ;
		*dbp++ = mx ;
		*dbp++ = mxy_size ;
		*dbp++ = t_idx ;
		*dbp++ = from_blk_size ;
		*dbp++ = md_z_len ;
#endif 

		fdp += blk_idx * from_blk_size ;	// adjust the from 
		// now ftp is pointing at the beginning of the from-block

		// starts from here ... same as single block func in version 1

		i = t_idx / record_size ;
		tdp += i * ( record_size + NUM_OF_HVT_INDEX ) ;	// beginning of record

		ip = ( int * ) tdp ;
		t = ( *ip++ ) & CUBE_INFO_T_MSK ;
		v = *ip++ ;
		h = *ip++ ;

		tdp += NUM_OF_HVT_INDEX ;

		t_idx %= record_size ;	// inside this record

		tt = t_idx / mxy_size ; // which frame

		if ( tt < md_z_len )
		{ 
			j = t_idx % mxy_size ;
			hh = j % mx ;	// which h
			vv = j / mx ;	// which v

			// from = t * cxy_size + ( v + vv ) * cx + h + hh ;
			// serial from = ( t + tt )  * mxy_size + ( v + vv ) * mx + h + hh ;
			from = ( t + tt )  * cxy_size + ( v + vv ) * cx + h + hh ;

#ifdef CUDA_OBS 
			*dbp++ = 2222222 ;

			*dbp++ = t ;
			*dbp++ = tt ;
			*dbp++ = i ;
			*dbp++ = t ;
			*dbp++ = v ;
			*dbp++ = h ;
			*dbp++ = t_idx ;
			*dbp++ = tt ;
			*dbp++ = hh ;
			*dbp++ = vv ;
			*dbp++ = from ;
			*dbp++ = fdp[ from ] ;
			*dbp++ = 99999999 ;
#endif 
			tdp[ t_idx ] = fdp[ from ] ; 
		} 
#ifdef CUDA_DBG 
		else
			tdp[ t_idx ] = 2222 ;
#endif 

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
h_do_motion_detection_step0_v2 ( T *fromp, T *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int record_size,	// do not includes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_xyzp,	// cube in device	// will have the size of the 
	int hvt_size,
	int from_block_size ) // new
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; //=  ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: fromp %p to %p size %d record %d mdxyz %d %d %d "
		"hvt %d from_block_size %d\n",
		__func__, fromp, top, tbl_size, record_size, md_x, md_y, md_z,
		hvt_size, from_block_size ) ;

	dbg_p_d_data_i ("xyz_table", ( int *)d_xyzp, CUBE_INFO_CNT * sizeof ( *d_xyzp ) / sizeof( int )) ;
#endif 

#ifdef CUDA_OBS 
	if (( tbl_size % number_blocks ) || ( tbl_size % cube_xy ) ||
		( record_size % cube_xy ))
	{
		fprintf(stderr, "%s: error size %d cube %d rec %d nblks %d\n",
			__func__, tbl_size, cube_xy, record_size, number_blocks ) ;
		return ( 0 ) ;
	}
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_motion_detection_step0_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		fromp, top,
		tbl_size,	// does not include the 3 indexes 
		record_size, // do not include the 3 indexes
		hvt_size,	// number of combination of h/v/t
		md_x, md_y, md_z,
		d_xyzp, from_block_size ) ;

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("motion_detect", top, tbl_size ) ; 
#endif 
	return ( 1 ) ;
}

template int
h_do_motion_detection_step0_v2<int> ( int *fromp, int *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int record_size,	// do not includes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_xyzp,	// cube in device	// will have the size of the 
	int hvt_size,
	int from_block_size ) ; // new

template int
h_do_motion_detection_step0_v2<float> ( float *fromp, float *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int record_size,	// do not includes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_xyzp,	// cube in device	// will have the size of the 
	int hvt_size,
	int from_block_size ) ; // new

// 3 indexes + real data length == record_length
// loopcnt ... number of records in each blk 
// tbl_size ... number of records in blk_x * blk_y blks
__global__ void d_do_motion_idx_v2 ( int *dp, int tbl_size, int loopcnt,
	int record_length,
	int h_loop, int t_loop, int hv_size,
	int blk_in_x, int blk_in_y ) 
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ot_idx, blk_idx, blk, i, j, k ;
	int *odp ;
	int *ip ;

#ifdef CUDA_OBS 
	if ( t_idx == 0 )
	{
		*dp++ = tbl_size ;
		*dp++ = loopcnt ;
		*dp++ = record_length ;
		*dp++ = h_loop ;
		*dp++ = t_loop ;
		*dp++ = hv_size ;
		*dp++ = blk_in_x ;
		*dp++ = blk_in_y ;
	}
#endif 

	ot_idx = t_idx ;
	odp = dp ;
	while ( t_idx < tbl_size )
	{
		blk_idx = t_idx / loopcnt ;	// which block
		dp = odp ;

		// j = ( t_idx % loopcnt ) % hv_size ;

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

		dp += t_idx * record_length ;

		t_idx -= blk_idx * loopcnt ;

		if ( t_idx == ( loopcnt - 1 ))
		{
			ip = ( int * ) dp ;

			*ip++ = CUBE_INFO_SET( blk ) ;	// tmporal

			*ip++ = (( hv_size / h_loop ) - 1 ) / 2 ;
			*ip++ = ( h_loop - 1 ) / 2 ;	 
		} else 
		{
			ip = ( int * ) dp ;
			*ip++ = (( t_idx / hv_size ) + 1 ) | CUBE_INFO_SET( blk ) ;	// tmporal

			j = t_idx % hv_size ;
	
			*ip++ = j / h_loop ;	// vertical
			*ip++ = j % h_loop ;	// horizontal
		}

		// dp += NUM_OF_HVT_INDEX ;	// assume float and int are of the same size ... QQQ ... FIX

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}  

// total size is the buffer size ... might not be the same as used.
// md_x/y/z aer each side
int
h_do_motion_idx_v2 ( int *dp, int total_size,
	int *orig_idx,
	int blk_in_x, int blk_in_y, struct cube *cubep,
	int md_x, int md_y, int md_z, int *record_sizep )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int record_length, k, i, nBlocks, loopcnt ;

	// the record length is the largest record amongst the inner/side/corner blks

	fprintf( stderr, "%s: total_size %d md x/y/z %d %d %d \n",
		__func__, total_size, md_x, md_y, md_z ) ;

	record_length = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		k = ( cubep[i].x - md_x * 2 ) *
			( cubep[i].y - md_y * 2 ) * ( cubep[i].z  - md_z + 1 ) ; 


#ifdef CUDA_DBG 
		fprintf(stderr, "%s: i %d cube %d %d %d k %d \n",
			__func__, i, cubep[i].x, cubep[i].y, cubep[i].z, k ) ;
#endif 
	
		if ( k > record_length )
			record_length = k ;
	}

	*record_sizep = record_length ;

	record_length += NUM_OF_HVT_INDEX ; // 3 indexes .. t/v/h in the beginning ...

	// the last record entry ( i.e. loopcnt -1 ) has a different format ...
	// ck the device code, when t_idx, ( loopcnt - 1)
	loopcnt = ( md_x * 2 + 1 ) * ( md_y * 2 + 1 ) * ( md_z - 1 ) + 1 ;

	i = record_length * loopcnt * blk_in_x * blk_in_y ;

	if ( i > total_size )
	{
		fprintf( stderr, "%s: size needed %d got %d\n",
			__func__, i, total_size ) ;
		return ( 0 ) ;
	}

	i /= record_length ;

	// nBlocks= ( i + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	fprintf( stderr, "%s: loopcnt %d i %d rec %d md %d %d %d blk x/y %d %d\n",
		__func__, loopcnt, i, record_length, md_x, md_y, md_z, blk_in_x,
		blk_in_y ) ;
 
	d_do_motion_idx_v2 <<< nBlocks, nThreadsPerBlock >>> (
		dp, i, loopcnt, record_length, ( md_x * 2 + 1 ), md_z, 
		( md_x * 2 + 1 ) * ( md_y * 2 + 1 ),
		blk_in_x, blk_in_y ) ;

	cudaThreadSynchronize() ;

	// *orig_idx = md_y * ( md_x * 2 + 1 ) + md_x ;
	*orig_idx = loopcnt - 1 ;

	return ( 1 ) ;
}

// step one is to get y0-yk
// no need to worry about the different cube size ... inner/side/corner
// it will be all junk anyway ...
template<typename T>
__global__ void d_do_l1_norm_step1_v2 ( T *dp, int tbl_size, int record_length,
	int orig, int hvt_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j ;
	T *odp, *op ;

	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		j = t_idx / record_length ;
		i = j / hvt_size ;	// index into the hvt_size size block
		j %= hvt_size ;	// record index into the current block

		if ( j != orig )
		{
			dp += i * hvt_size * ( record_length + NUM_OF_HVT_INDEX ) ; 
			op = dp + orig * ( record_length + NUM_OF_HVT_INDEX ) + NUM_OF_HVT_INDEX ;

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

// now make all entries positive for orig
// table_size is the all the entries of all orig in all ( nblk_in_x * nblk_in_y ) block

template<typename T>
__global__ void d_do_l1_norm_step1_1_v2 ( T *dp, int tbl_size, int record_length,
	int orig, int hvt_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j ;
	T *odp ;

	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		j = t_idx / record_length ;	// which hvt_size block 
		dp += ( j * hvt_size + orig ) * ( record_length + NUM_OF_HVT_INDEX ) +
			( t_idx % record_length ) + NUM_OF_HVT_INDEX ;

		if ( *dp < 0 )
			*dp = -*dp ;	// save a step ... no need to abs()

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// total and record_size does not have the NUM_OF_HVT_INDEX elements
template<typename T>
int
h_do_l1_norm_step1_v2( T *dp, int total, int record_size, int orig, int hvt_size)
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( total + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	if (( total % record_size ) || ( total % ( hvt_size * record_size )))
	{
		fprintf( stderr, "%s: error total %d rec %d hvt %d \n",
			__func__, total, record_size, hvt_size ) ;
		return ( 0 ) ; 
	}

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step1_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, total, record_size, orig, hvt_size ) ;

	cudaThreadSynchronize() ;

	total = ( total / ( record_size * hvt_size )) * record_size ;

	// total = nblk_x * nbli_y * record_size

	// nBlocks = ( total + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( total, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step1_1_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, total, record_size, orig, hvt_size ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;

}

template int
h_do_l1_norm_step1_v2<int>( int *dp, int total, int record_size, int orig, int hvt_size) ;

template int
h_do_l1_norm_step1_v2<float>( float *dp, int total, int record_size, int orig, int hvt_size) ;

// step two is to get L1-norm(sum)
// all row, should be after the abs() is done
// tbl_size is the number of elements for this addition operation
// record_length includes the NUM_OF_HVT_INDEX
// cnt is the max_cnt for each record, regardless inner/side/corner

template<typename T>
__global__ void d_do_l1_norm_step2_v2 ( T *dp, int tbl_size, int record_length,
	int cnt, struct cube *d_xyzp, int *d_resp )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *ip, record_type, start, current_cnt , j ;
	long long l, ll ;
	T *fp, *odp ;

	odp = dp ;
	while ( t_idx < tbl_size )
	{
		dp = odp ;

		j = t_idx / cnt ;

		dp += record_length * j ;

		ip = ( int * )dp ;

		record_type = CUBE_INFO_GET( *ip ) ;

		start = d_xyzp[ record_type ].z ; 
		current_cnt = d_xyzp[ record_type ].y ; 

		j = t_idx % cnt ;

		if ( current_cnt > j )
		{
			dp += NUM_OF_HVT_INDEX ;
			fp = dp + start ;

			// this check is needed when *dp is int
			{
				l = dp[ j ] ;
				ll = fp [ j ] ;

				l += ll ;
	
				if ( l & 0xffffffff00000000 )
					*d_resp = t_idx ;
			}

			dp[ j ] += fp [ j ] ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// step 1.1 should be the abs() ... not needed, done in step 1 
// step 2 is to do the sum
// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
// NOTE d_xyzp->y/z will be destroyed ... x
// hcubep: has been adjusted to the after md_x/y/z size ; 
template<typename T>
int
h_do_l1_norm_step2_v2( T *dp, int total, int record_size,
	struct cube *hcubep, struct cube *d_xyzp, int *d_resp )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks, i, start, row, cnt ;
	int max_cnt ;
	struct cube cxyz[3] ;

#ifdef CUDA_OBS
	fprintf( stderr, "%s: dp %p total %d record %d \n",
		__func__, dp, total, record_size ) ;
#endif 

	max_cnt = 0xdeadbeef ;

	if ( !put_d_data_i ( d_resp, &max_cnt, sizeof ( int )))
	{
		fprintf( stderr, "%s: put data failed d_resp %p \n", __func__, d_resp ) ;
	}

	memcpy ( cxyz, hcubep, sizeof ( cxyz )) ;

	max_cnt = 0 ;
	for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
	{
		// y is cnt ... z is start ...

		cnt = cxyz[i].y = hcubep[i].x * hcubep[i].y * hcubep[i].z ;	// size 

		start = max_log2( cnt ) ;
		if ( start != cnt )
			start = max_log2(( start / 2 ) - 1 ) ;
		else
			start >>= 1 ;
		
		cxyz[i].z = start ;
		cxyz[i].y -= start ;

		if ( max_cnt < cxyz[i].y )
			max_cnt = cxyz[i].y ;

#ifdef CUDA_DBG
		fprintf( stderr, "%s: i %d z %d y %d max %d cnt %d \n",
			__func__, i, cxyz[i].z, cxyz[i].y, max_cnt, cnt ) ;
#endif 
	}

	h_set_cube_config ( d_xyzp, cxyz ) ; 

	row = total / record_size ;

	if ( total % record_size )
	{
		fprintf( stderr, "h_do_l1_norm_step2_v2: error size %d %d \n", total, record_size ) ;
		return ( 0 ) ;
	}

	while ( max_cnt > 0 ) 
	{
		i = row * max_cnt ;

#ifdef CUDA_DBG 
		fprintf( stderr, "row %d cnt %d i %d \n", row, max_cnt, i ) ;
#endif 
		
		// nBlocks= ( i + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

		h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step2_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
			dp, i, record_size + NUM_OF_HVT_INDEX,
			max_cnt, d_xyzp, d_resp ) ;

		cudaThreadSynchronize() ;

		if ( !get_d_data_i ( d_resp, &i, sizeof ( int )))
		{
			fprintf( stderr, "%s: get data failed \n", __func__ ) ;
		}

		if ( i != 0xdeadbeef )
			fprintf( stderr, "%s: overflow error return %x \n", __func__, i ) ;

		max_cnt = 0 ;
		for ( i = 0 ; i < CUBE_INFO_CNT ; i++ )
		{
			cxyz[i].z >>= 1 ;
			cxyz[i].y = cxyz[i].z ;

			if ( max_cnt < cxyz[i].y )
				max_cnt = cxyz[i].y ;

#ifdef CUDA_DBG
		fprintf(stderr,"%s-2: i %d z %d y %d max %d \n",
			__func__, i, cxyz[i].z, cxyz[i].y, max_cnt ) ;
#endif 
		}
		h_set_cube_config ( d_xyzp, cxyz ) ; 
	}

	h_set_cube_config ( d_xyzp, hcubep ) ; 

	return ( 1 ) ;
}

template<> int
h_do_l1_norm_step2_v2<int>( int *dp, int total, int record_size,
	struct cube *hcubep, struct cube *d_xyzp, int *d_resp ) ;

template int
h_do_l1_norm_step2_v2<float>( float *dp, int total, int record_size,
	struct cube *hcubep, struct cube *d_xyzp, int *d_resp ) ;

#define MAX_L1_NORM			1000

// step 3 is to get 1-|y0-yk|/|y0| 
// row_size is the number of rows ... 
// record_length includes the NUM_OF_HVT_INDEX
// dp starts with valid data, see caller

template<typename T>
__global__ void d_do_l1_norm_step3_v2 ( T *dp, int row_size, int record_length,
	int orig, int hvt_size )
{
	int ot_idx, i, t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *op, *odp ;

	odp = dp ;
	ot_idx = t_idx ;
	while ( t_idx < row_size )
	{
		dp = odp ;

		i = t_idx / hvt_size ;	// which block
		t_idx -= i * hvt_size ;	// which record in block

		// skip the orig
		if ( t_idx != orig )
		{
			dp += ( i * hvt_size ) * record_length ;
			op = dp + orig * record_length ;
			dp += t_idx * record_length ;
			
			// FIX ... if no int ... then there is no such problem

			// *dp = MAX_L1_NORM - ( MAX_L1_NORM * ( *dp )) / (*op) ;
			*dp = ((T)MAX_L1_NORM) - ( T )((( float )MAX_L1_NORM ) * ((( float )*dp ) / (( float ) *op ))) ;
		}

		ot_idx += CUDA_MAX_THREADS ;
		t_idx = ot_idx ;
	}   
}

// record_size does not have the NUM_OF_HVT_INDEX elements
// total is the overall number of data elements, no NUM_OF_HVT_INDEX
template<typename T>
int
h_do_l1_norm_step3_v2( T *dp, int total, int record_size, int orig, int hvt_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int i, nBlocks ;

	if ( total % record_size )
	{
		fprintf( stderr, "h_do_l1_norm_step3_v2: error size %d %d \n", total, record_size ) ;
		return ( 0 ) ;
	}

	i = total / record_size ;

	if ( i % hvt_size )
	{
		fprintf( stderr, "h_do_l1_norm_step3_v2: error row %d hvt %d \n", i, hvt_size ) ;
		return ( 0 ) ;
	}

	// nBlocks= ( i + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( i, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step3_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp + NUM_OF_HVT_INDEX, i, record_size + NUM_OF_HVT_INDEX,
		orig, hvt_size ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

template int
h_do_l1_norm_step3_v2<int>( int *dp, int total, int record_size, int orig, int hvt_size ) ;

template int
h_do_l1_norm_step3_v2<float>( float *dp, int total, int record_size, int orig, int hvt_size ) ;

// to find the max of each hvt_size block
// record_length includes the NUM_OF_HVT_INDEX
// dp starts before the NUM_OF_HVT_INDEX,  see caller
// total is the number of hvt_blocks

template<typename T>
__global__ void d_do_l1_norm_step4_3_v2 ( T *dp, int total, int record_length,
	int hvt_size, int orig )
{
	int *fip, *tip, t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	T *odp, *fp ;

	odp = dp ;
	while ( t_idx < total )
	{
		dp = odp ;

		fp = dp + t_idx * record_length * hvt_size ;
		dp += t_idx * L1_NORM_STEP4_RETURN_ENTRY_SIZE * 2 ;	// first is the max, secondis no motion

		tip = ( int *)dp ;
		fip = ( int *)fp ;

		*tip++ = *fip++ & CUBE_INFO_T_MSK ; 
		*tip++ = *fip++ ; 
		*tip++ = *fip++ ; 
		*tip++ = *fip ; 

		fp = odp + ( t_idx * hvt_size + orig ) * record_length ;
		fip = ( int *)fp ;

		*tip++ = *fip++ & CUBE_INFO_T_MSK ; 
		*tip++ = *fip++ ; 
		*tip++ = *fip++ ; 
		*tip++ = *fip ; 

		t_idx += CUDA_MAX_THREADS ;
	}   
}

// to find the max of each hvt_size block
// record_length includes the NUM_OF_HVT_INDEX
// dp starts with idx 0 after NUM_OF_HVT_INDEX, see caller
// cnt is the number of rows need to be processed in this hvt_size rows
// total is the number of entries in ( hvt_size * cnt ) need to be 
//		processed at this run

template<typename T>
__global__ void d_do_l1_norm_step4_2_v2 ( T *dp, int total, int record_length,
	int start, int cnt, int hvt_size )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, *tip, *fip ;
	T *tp, *odp ;

	odp = dp ;
	while ( t_idx < total )
	{
		dp = odp ;

		i = t_idx / cnt	; // number of the hvt_size block	
		dp += i * hvt_size * record_length ;	
			// dp points to the first row in this hvt_size block

		i = t_idx % cnt	; // index into this hvt_size block after start

		tp = dp + i * record_length ;	// destination
		dp = tp + start * record_length ;

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

// step 4.1: move the no_motion_row to the orig ... 
// total is hvt_size * blk_in_x * blk_in_y i.e. total is the number of records
// record_length has the NUM_OF_HVT_INDEX
// dp points to the correct data space behind NUM_OF_HVT_INDEX
// no_motion_idx is the block right after the orig in t-domain and no
//	shift in the h/v direction

template<typename T>
__global__ void d_do_l1_norm_step4_1_v2 ( T *dp, int total, int record_length,
	int orig, int hvt_size, int no_motion_idx )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int *tip, *fip, i ;
	T *tdp, *odp ;

	odp = dp ;
	while ( t_idx < total )
	{
		dp = odp ;

		i = t_idx % hvt_size ;	// which record in the block

		dp += t_idx * record_length ;

		if ( i != orig )
		{
			if ( i == no_motion_idx )
			{
				// ok now the no motion one is in the orig "row" ...
				// this is the orig row ... 
				tdp = dp + ( orig - no_motion_idx ) * record_length ;

				tip = ( int *)tdp ;
				fip = ( int *)dp ;

				*tip-- = *fip-- ; 	// value
				*tip-- = *fip-- ; 	// h
				*tip-- = *fip-- ; 	// v
				*tip = *fip ; 	// t
#ifdef CUDA_OBS 	 // let the smallest negative number, when all numbers are negative, wins.
			} else if ( *dp < 0 )
				*dp = 0 ;
#else
			}
#endif 
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

// total is overall data area
// record_size does not include NUM_OF_HVT_INDEX
// orig: the block that every "moving" blocks compared with
template<typename T>
int
h_do_l1_norm_step4_v2( T *dp, int total, int record_size, int orig,
	int hvt_size, int *resp, int no_motion_idx )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int blocks, nBlocks, i, start, row, cnt ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: dp %p total %d rec %d orig %d hvt %d resp %p\n",
		__func__, total, total, record_size, orig, hvt_size, resp ) ;
#endif 

	if ( total % record_size )
	{
		fprintf( stderr, "%s: error size %d %d \n", __func__, total,
			record_size ) ;
		return ( 0 ) ;
	}

	row = total / record_size ;

	if ( row % hvt_size )
	{
		fprintf( stderr, "%s: error hvt %d row %d \n", __func__,
			hvt_size, row ) ;
		return ( 0 ) ;
	}

	blocks = row / hvt_size ; // i.e. blk_in_x * blk_in_y

	// step 4.1 ... 0 out all orig and negative entries ...

	// nBlocks= ( row + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( row, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step4_1_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp + NUM_OF_HVT_INDEX, row,
		record_size + NUM_OF_HVT_INDEX, orig, hvt_size, no_motion_idx ) ;

	cudaThreadSynchronize() ;

	printf("%s : step 4.1 done \n", __func__ ) ;

	// step 4.2 ... get the max 

	start = max_log2( hvt_size ) ;
	if ( start != hvt_size )
		start = max_log2(( start / 2 ) - 1 ) ;
	else
		start >>= 1 ;
	
	cnt = hvt_size - start ;

	while ( cnt > 0 ) 
	{
#ifdef CUDA_DBG 
		printf("%s : row %d cnt %d start %d\n", __func__, row, cnt, start ) ;
#endif 

		// nBlocks= ( cnt * blocks + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

		h_block_adj ( cnt * blocks, nThreadsPerBlock, &nBlocks ) ;

		d_do_l1_norm_step4_2_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
			dp + NUM_OF_HVT_INDEX, cnt * blocks, record_size + NUM_OF_HVT_INDEX,
			start, cnt, hvt_size ) ;

		cudaThreadSynchronize() ;

		start >>= 1 ;
		cnt = start ;
	}

	printf("%s : step 4.2 done \n", __func__ ) ;

	// nBlocks = ( blocks + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( blocks, nThreadsPerBlock, &nBlocks ) ;

	d_do_l1_norm_step4_3_v2<T> <<< nBlocks, nThreadsPerBlock >>> (
		dp, blocks, record_size + NUM_OF_HVT_INDEX, hvt_size, orig ) ;

	cudaThreadSynchronize() ;

	// 2: is the t/v/h/value for best one and the no move one
	if (( blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2 ) > (( record_size +
		NUM_OF_HVT_INDEX ) * hvt_size ))
	{
		fprintf(stderr, "%s: error: size mismatch %ld %ld\n", __func__, 
			(unsigned long)( blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2 ),
			(unsigned long)(( record_size + NUM_OF_HVT_INDEX ) * hvt_size )) ;
		return ( 0 ) ;
	}

	printf("%s : step 4.3.1 done \n", __func__ ) ;
	printf("%s: outbuf %p device %p blks %d size %ld\n", __func__,
	       resp, dp, blocks,
	       (unsigned long)(blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2 )) ;

	if (( i = cudaMemcpy( resp, dp,
		blocks * L1_NORM_STEP4_RETURN_ENTRY_SIZE * sizeof( int ) * 2, 	
			// one for max, one for no motion
		cudaMemcpyDeviceToHost)) != cudaSuccess )
	{
		fprintf(stderr, "%s: memcpy failed %d\n", __func__, i ) ;
		return ( 0 ) ;
	}

	printf("%s : step 4.3 done \n", __func__ ) ;

	return ( 1 ) ;
}

template int
h_do_l1_norm_step4_v2<int>( int *dp, int total, int record_size, int orig,
	int hvt_size, int *resp, int no_motion_idx ) ;

template int
h_do_l1_norm_step4_v2<float>( float *dp, int total, int record_size, int orig,
	int hvt_size, int *resp, int no_motion_idx ) ;
