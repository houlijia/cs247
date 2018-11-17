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

#include "cs_vector.h"

#define CUDA_DBG
#define CUDA_DBG1

__global__ void d_do_vector_zero_some ( float *in1, int *tblp, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		in1[ tblp[ t_idx ]] = 0 ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

/*
h_do_vector_add_destroy: will zero out the data fields in the vector d_datap,
these fields are indexed by
the entries in the d_inp.  the size of d_inp is tbl_size.
*/

void
h_do_vector_zero_some ( float *d_datap, int *d_inp, int tbl_size )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_vector_zero_some <<< nBlocks, nThreadsPerBlock >>> ( d_datap, d_inp, tbl_size ) ;

	cudaThreadSynchronize() ;
}
