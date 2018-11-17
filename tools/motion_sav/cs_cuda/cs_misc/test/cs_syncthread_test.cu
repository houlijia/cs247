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

#define A_SIZE	4

struct rec {
	int f ;
	int idx ;
	int a[A_SIZE] ;
} ;

#define CUDA_DBG 

#define NUM_BUF	2500

#define THR_PER_BLOCK	512

struct rec hbuf[NUM_BUF], hbuf1[NUM_BUF] ;
struct rec *dbuf ;

void p_data( char *s, struct rec *bp, int size ) ;

__global__ void 
d_cmp ( struct rec *odp, int str, int total )
{
#ifdef CUDA_OBS 
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
#endif 
	int start_idx, k, size, start, cnt ;
	struct rec *tp, *fp ;
#ifdef CUDA_DBG 
	int done_one = 0 ;
#endif 

	// the first record of this blk ( i.e. blockDim.x * 2 )
	start_idx = blockIdx.x * blockDim.x * 2 ;	

	k = total - start_idx ;		// how many record left?

	size = ( k > blockDim.x * 2 ) ? blockDim.x * 2 : k ;

#ifdef CUDA_OBS 
	if ( t_idx == 8 )
	{
		odp->f = k ;
		odp->idx = size ;
		odp->a[0] = blockDim.x ;
		odp->a[1] = blockIdx.x ;
		odp->a[2] = threadIdx.x ;
		odp->a[3] = total ;

	}
#endif 

#ifdef CUDA_DBG 
	while ( size > 1 )
	{
		start = ( size + 1 ) / 2 ;

		cnt = size - start ;

		if ( !done_one )
		if ( threadIdx.x < cnt ) 
		{
			tp = odp + ( start_idx + threadIdx.x ) * str ;	// destination
			fp = tp + start * str ;   // src

			if ( tp->f < fp->f )
			{
				tp->f = fp->f ;
				tp->idx = fp->idx ;
			}
		}

		size -= cnt ;

		// t_idx += CUDA_MAX_THREADS ; // do this before calling __syncthreads()

		__syncthreads() ; // wait for all threads in this block

		// done_one++ ;
	}   
#endif 
}

void
h_cmp( struct rec *dp, int size )
{
	int nBlocks, nThreadsPerBlock = THR_PER_BLOCK ;
	int skip_size, cnt, stride = 1 ;
#ifdef CUDA_DBG 
	int done_one = 0 ;
#endif 

	// START HERE ...

	cnt = size ;
	skip_size = nThreadsPerBlock * 2 ;

	while ( cnt != 1 ) 
	{
		h_block_adj (( cnt + 1 )/2, nThreadsPerBlock, &nBlocks ) ;

		printf("%s : cnt %d stride %d blks %d \n", __func__, cnt, stride, nBlocks ) ;

		d_cmp <<< nBlocks, nThreadsPerBlock >>> ( dp, stride, cnt ) ;

		cudaThreadSynchronize() ;

		stride *= skip_size ;

		cnt = ( cnt + skip_size - 1 ) / skip_size ; 

		printf("%s : last cnt %d stride %d\n", __func__, cnt, stride ) ;
#ifdef CUDA_DBG 
		done_one++ ;

		dbg_get_d_data(( char *) dp, ( char *)hbuf1, sizeof ( hbuf1 )) ; 
		p_data( "===== DONE ", hbuf1, NUM_BUF ) ;

#ifdef CUDA_OBS 
		if ( done_one == 2 )
			cnt = 1 ;
#endif 
#endif 
	}
}

int
make_data()
{
	int max_idx, max, k, i ;

	if (( k = cudaMalloc( &dbuf, sizeof( struct rec ) * NUM_BUF )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	max_idx = 0 ;
	max = -999999 ;
	for ( i = 0 ; i < NUM_BUF ; i++ )
	{
		hbuf[i].idx = i ;
		k = rand() ;
		hbuf[i].f = k ;

		if ( k > max )
		{
			max_idx = i ;
			max = k ;
		}

		for ( k = 0 ; k < A_SIZE ; k++ )
			hbuf[i].a[k] = ( i << 16 ) | k ;
	}

	// hbuf[0].f = max + 1 ; // test for the first ...
	// hbuf[1].f = max + 1 ; // test for the 2nd from the first ...
	// hbuf[NUM_BUF-1].f = max + 1 ; // test for the last ...
   	// hbuf[NUM_BUF-2].f = max + 1 ; // test for the 2nd to last  ...

	dbg_put_d_data(( char *) dbuf, ( char *)hbuf, sizeof ( hbuf )) ; 

	p_data( "before", hbuf, NUM_BUF ) ;

	printf("max is %d idx %d\n", max, max_idx ) ;

	return ( 1 ) ;
}

main()
{
	if ( !make_data())
		exit( 3 ) ;
	
	h_cmp( dbuf, NUM_BUF ) ;

	dbg_get_d_data(( char *) dbuf, ( char *)hbuf1, sizeof ( hbuf1 )) ; 

	p_data( "after", hbuf1, NUM_BUF ) ;
}

void
p_data( char *s, struct rec *bp, int size )
{
	int i ;

	printf("%s ::: %s ------------------------------------------------------ \n", __func__, s ) ;

	for ( i = 0 ; i < size ; i++ )
	{
		printf("%d -- idx %d	v %d	a %x	%x	%x	%x\n",
			i, bp->idx, bp->f, bp->a[0], bp->a[1], bp->a[2], bp->a[3] ) ;
		bp++ ;

		if (!(( i + 1 ) % THR_PER_BLOCK ))
			printf("\n") ;
	}
	printf("%s ::: END ------------------------------------------------------ \n", __func__ ) ;
}





