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
#include "cs_cuda.h"
#include "cs_helper.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_random.h"

// #define CUDA_DBG
#define CUDA_DBG1

static RndCState cs_random_state ;

// LDL stop here ... in SensingMatrixWH.makepuermutations
// PL = 1 .. 1 + random(order - 1) 
// PR = random(order) 


// val can be either positive or negative

__global__ void 
d_set_random_table( RndC_uint32 *fp, int size, RndC_uint32 val )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	while ( tid < size )
	{
		fp[ tid ] += val ;
		tid += CUDA_MAX_THREADS ;
	} 
}

/*
    seed : the seed for random number generation
	h_in: points to the host buffer before download to device, if null, a temp buffer will be
		allocated and deleted when done.
	slot_one_reserved: if set [0] will have 0
*/

int
h_set_random_table ( RndC_uint32 seed, RndC_uint32 *h_in, RndC_uint32 *d_out, int size,
	int slot_one_reserved, int do_init )	
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ;
	RndC_uint32 *h_in_memp = NULL ;
	int i, need_adjustment = 0 ;

	if ( h_in == NULL )
	{
		h_in_memp = ( RndC_uint32 *)malloc ( sizeof ( RndC_uint32 ) * size ) ;
		if ( h_in_memp == NULL )
			return ( 0 ) ;
	} else
		h_in_memp = h_in ;


	if ( do_init )
		init_RndC( &cs_random_state, seed ) ;

	if ( slot_one_reserved )
	{
		h_in_memp[0] = ( RndC_uint32 )0 ;
		randperm1_RndC( &cs_random_state, size - 1, h_in_memp+1 ) ; // from 1 to ( size - 1 )
	} else
	{
		randperm1_RndC( &cs_random_state, size, h_in_memp ) ; // from 1 to ( size )
			// need to subtract one 
		need_adjustment++ ;
	}

	if (( i = cudaMemcpy( d_out, h_in_memp, sizeof ( RndC_uint32 ) * size,
		cudaMemcpyHostToDevice)) != cudaSuccess )
	{
		printf("%s: download cp fail: %d\n", __func__, i ) ;
		
		if ( h_in == NULL )
			free ( h_in_memp ) ;
		return ( 0 ) ;
	}

	if ( need_adjustment )
	{
		h_block_adj ( size, nThreadsPerBlock, &nBlocks ) ;

		d_set_random_table <<< nBlocks, nThreadsPerBlock >>> ( d_out, size, ( RndC_uint32) -1 ) ;

		cudaThreadSynchronize() ;
	}

	if ( h_in == NULL )
		free ( h_in_memp ) ;

	return ( 1 ) ;
}

