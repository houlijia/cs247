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

#define INC_SIZE	17
#define MOD_SIZE	100

#define BLK_DIM		32

#define DATA_SIZE0	1
#define DATA_SIZE1	3

#define DATA_NUM	2 // one for size 0, one for size1
#define DATA_MAX	(( DATA_SIZE0 > DATA_SIZE1 ) ? DATA_SIZE0 : DATA_SIZE1 )
	 // max of the DATA_SIZE?

int buf[ DATA_NUM * DATA_MAX ] ;

static int *d_memp, *d_dbgp ;

__global__ void 
d_size_test ( int *dmemp, int rec_size, int *dbgp )
{
	int total_size ;
	int i, start, blk_size ;
	int last_chunk, k ;
	int *fp, *tp ;

	blk_size = blockDim.x ;

	if ( blockIdx.x == 0 )
		total_size = DATA_SIZE0 ;
	else
		total_size = DATA_SIZE1 ;

	// record size is fixed ...
	dmemp += blockIdx.x * rec_size ; 	// dmemp points to the beginning of this record

	k = ( total_size + blk_size - 1 ) / blk_size ;
	last_chunk = total_size % blk_size ;

	if ( last_chunk )
		k-- ;

	for ( i = 1 ; i <= k ; i++ )
	{
		tp = dmemp ;
		fp = tp + i * blk_size ;	// first chunk
		if ( i < k )
			tp[ threadIdx.x ] += fp [ threadIdx.x ] ;
		else if ( threadIdx.x < last_chunk )
			tp[ threadIdx.x ] += fp [ threadIdx.x ] ;

		__syncthreads() ; // wait for all threads in this block

	}

	// return ; // TTT

	if ( total_size > blk_size )
		total_size = blk_size ;

	start = blk_size / 2 ;	// either 1/2 of DATA_MAX ( if DATA_MAX > BLK_DIM )
		// or max.mod_log2 of data ... ex. if data is 31, it is 16
	last_chunk = total_size - start ;

	while ( total_size > 1 )
	{
		if ( last_chunk > 0 )
		{
			tp = dmemp ;
			fp = dmemp + start ;

			if ( threadIdx.x < last_chunk )
				tp[ threadIdx.x ] += fp [ threadIdx.x ] ;
		}
			
		__syncthreads() ; // wait for all threads in this block

		if ( last_chunk > 0 )
			total_size -= last_chunk ;

		start >>= 1 ;
		last_chunk = total_size - start ;
	}
}

main( int ac, char *av[] )
{
	int *ip, i, j ;
	dim3	grid, blk ;

	ip = buf ;
	j = 0 ;
	i = DATA_NUM * DATA_MAX ;
	while ( i-- )
	{
		*ip++ = ++j ;

		if ( i == DATA_MAX )
			j = 0 ;
	}
	// dbg_pdata_i("H BUF", buf, DATA_NUM * DATA_MAX ) ; 
		
	dbg_init( 3000000 ) ;

	if (( i = cudaMalloc( &d_memp, DATA_NUM * DATA_MAX * sizeof( int ))) != cudaSuccess )
	{
		printf("%s: dbg cudaMalloc failed %d\n", __func__, i ) ;
		exit ( 3 ) ;
	}

	if (( i = cudaMalloc( &d_dbgp, DATA_NUM * DATA_MAX * sizeof( int ))) != cudaSuccess )
	{
		printf("%s: dbg cudaMalloc 2 failed %d\n", __func__, i ) ;
		exit ( 3 ) ;
	}
	printf("d_dbgp %p d_memp %p\n", d_dbgp, d_memp ) ; 

	dbg_put_d_data((char *)d_memp, (char *)buf, DATA_NUM * DATA_MAX * sizeof ( int )) ;

	set_device_mem_i ( d_dbgp, DATA_NUM * DATA_MAX, 0 ) ;

	grid.x = DATA_NUM ;	// the two is the number of row, SIZE0 and SIZE1 
	grid.y = 1 ;
	grid.z = 1 ;

	// block size ... i.e. 32 ... in real, either 512 or 1024
	// blk.x = ( BLK_DIM > DATA_MAX ) ? DATA_MAX : BLK_DIM ;
	blk.x = BLK_DIM ;
	blk.y = 1 ;
	blk.z = 1 ;

	d_size_test <<< grid, blk >>> ( d_memp, DATA_MAX, d_dbgp ) ; // the max in all DATA_SIZE?
		// data

	cudaThreadSynchronize() ;

	dbg_p_d_data_i("SIZE ON DEVICE", d_memp, DATA_MAX * DATA_NUM ) ; 
	//p dbg_p_d_data_i("SIZE CKING", d_dbgp, DATA_MAX * DATA_NUM ) ; 

	exit(0);
}
