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

// #define CUDA_DBG
// #define CUDA_DBG1

/* size is in 2^size */
int
permutation_load( int size, char *dirp, int *d_leftp,
	int *d_rightp )
{
	int i, ret = 0, cnt, fd = -1 ;
	char *fn ;
	int memsize, *leftp, *rightp, *lp, *rp ;

	leftp = rightp = NULL ;
	fn = NULL ;

	cnt = strlen( dirp ) ;
	if ( !( fn = ( char * )malloc ( cnt + 100 )))
	{
		printf("%s: malloc failed fn\n", __func__ ) ;
		goto done_err ;
	}

	sprintf( fn, "%s/WH_lfsr_%d_indcs.dat", dirp, size ) ;

#ifdef CUDA_DBG 
	printf("%s: filename \"%s\", size %d\n", __func__, fn, size ) ;
#endif 

	fd = open ( fn, O_RDONLY ) ;

	if ( fd < 0 )
	{
		printf("%s: %s open failed %d\n", __func__, fn, errno ) ;
		goto done_err ;
	}

	size = ( int )pow( 2.0, ( double )size ) ;

	memsize = size * sizeof( *leftp ) ;

	if (!( leftp = ( int * )malloc( memsize )))
	{
		printf("%s: malloc failed left\n", __func__ ) ;
		goto done_err ;
	}

	if (!( rightp = ( int * )malloc( memsize )))
	{
		printf("%s: malloc failed right\n", __func__ ) ;
		goto done_err ;
	}

	lp = leftp ;
	rp = rightp ;

#ifdef CUDA_DBG 
	printf("size %d\n", size ) ;
#endif 

	*lp++ = 0 ;
	*rp++ = 0 ;

	cnt = 1 ;
	size-- ;
	while ( size-- )
	{
		if ( read( fd, lp, sizeof ( *lp )) != sizeof( *lp ))
		{	
			printf("not enough cnt %d size %d\n", cnt, size ) ;
			goto done_err ;
		}

		if ( read( fd, rp, sizeof ( *rp )) != sizeof( *rp ))
		{	
			printf("not enough cnt %d size %d\n", cnt, size ) ;
			goto done_err ;
		}

		*lp = ntohl ( *lp ) ;
		*rp = ntohl ( *rp ) ;

#ifdef CUDA_OBS 
		printf("%d : l %x -- %d == %x	%d\n", cnt, *lp, *lp, *rp, *rp ) ;
#endif 

		cnt++ ;
		rp++ ;
		lp++ ;
	}

	if (( i = cudaMemcpy( d_leftp, leftp, memsize, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		printf("%s: download L fail: %d\n", __func__, i ) ;
		goto done_err ;
	}

#ifdef CUDA_OBS 
	dbg_p_d_data_i("perm_tbl_left", d_leftp, memsize >> 2 ) ; 
#endif 

	if (( i = cudaMemcpy( d_rightp, rightp, memsize, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		printf("%s: download R fail: %d\n", __func__, i ) ;
		goto done_err ;
	}

#ifdef CUDA_OBS 
	dbg_p_d_data_i("perm_tbl_right", d_rightp, memsize >> 2 ) ; 
#endif 

	ret = 1 ;

done_err:
	if ( rightp )
		free ( rightp ) ;

	if ( leftp )
		free ( leftp ) ;

	if ( fn )
		free ( fn ) ;

	if ( fd >= 0 )
		close( fd ) ;

	return ( ret ) ;
}

__global__ void d_do_permutation_R ( int *input, int *output,
	int *idxp, int size, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j ;

	while ( t_idx < size )
	{
		i = t_idx % tbl_size ;
		j = ( t_idx / tbl_size ) * tbl_size ;
		output[ j + idxp[ i ]] = input[ t_idx ] ;  

		
		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_permutation_R ( int *d_input, int *d_output, int *d_perm_tbl,
	int n, int tbl_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( n + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p size %d tblsize %d\n", __func__,
		d_input, d_output, d_perm_tbl, n, tbl_size ) ;
#endif 

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_R <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl, n, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("Right perm before", d_input, n ) ; 
	dbg_p_d_data_i("Right perm after", d_output, n ) ; 
#endif 
}
 
__global__ void d_do_permutation_Lv2 ( int *input, int *output,
	int *idxp_i, int *idxp_s, int *idxp_c, int size, int tbl_size,
	int nblk_in_x, int nblk_in_y )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j, *idxp ;

	while ( t_idx < size )
	{
		i = t_idx / tbl_size ;

		j = i / nblk_in_x ; 	// 0 .. ( nblk_in_y - 1 ) 
		i = i % nblk_in_x ;		// 0 .. ( nblk_in_x - 1 )

		if ( i == 0 )
		{
			if (( j == 0 ) || ( j == ( nblk_in_y  - 1 )))
				idxp = idxp_c ;
			else
				idxp = idxp_s ;
		} else if ( i == ( nblk_in_x - 1 ))
		{
			if (( j == 0 ) || ( j == ( nblk_in_y  - 1 )))
				idxp = idxp_c ;
			else
				idxp = idxp_s ;
		} else
		{
			if (( j == 0 ) || ( j == ( nblk_in_y  - 1 )))
				idxp = idxp_s ;
			else
				idxp = idxp_i ;
		}

		i = t_idx % tbl_size ;
		j = ( t_idx / tbl_size ) * tbl_size ;

		output[ t_idx ] = input[ j + idxp[ i ]] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_permutation_Lv2 ( int *d_input, int *d_output,
	int *d_perm_tbl_i, int *d_perm_tbl_s, int *d_perm_tbl_c, int n, int tbl_size,
	int nblk_in_x, int nblk_in_y )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( n + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p %p %psize %d tblsize %d\n", __func__,
		d_input, d_output, d_perm_tbl_i, d_perm_tbl_s, d_perm_tbl_c,
		n, tbl_size ) ;
#endif 

#ifdef CUDA_OBS 
	dbg_p_d_data_i("Left inner", d_perm_tbl_i, tbl_size ) ; 
	dbg_p_d_data_i("Left side", d_perm_tbl_s, tbl_size ) ; 
	dbg_p_d_data_i("Left corner", d_perm_tbl_c, tbl_size ) ; 
#endif 

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_Lv2 <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl_i, d_perm_tbl_s, d_perm_tbl_c, n, tbl_size,
		nblk_in_x, nblk_in_y ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("Left perm before", d_input, n ) ; 
	dbg_p_d_data_i("Left perm after", d_output, n ) ; 
#endif 
}

__global__ void d_do_permutation_L ( int *input, int *output,
	int *idxp, int size, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j ;

	while ( t_idx < size )
	{
		i = t_idx % tbl_size ;
		j = ( t_idx / tbl_size ) * tbl_size ;

		output[ t_idx ] = input[ j + idxp[ i ]] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_permutation_L ( int *d_input, int *d_output,
	int *d_perm_tbl, int n, int tbl_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( n + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	printf("%s: din %p dout %p perm %p size %d tblsize %d\n", __func__,
		d_input, d_output, d_perm_tbl, n, tbl_size ) ;
#endif 

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_L <<< nBlocks, nThreadsPerBlock >>> ( d_input,
		d_output, d_perm_tbl, n, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("Left perm before", d_input, n ) ; 
	dbg_p_d_data_i("Left perm after", d_output, n ) ; 
#endif 
}

/*
the 2nd permutation table is used for per block purpose ... only
load once for testing purpose
*/
/* size is in byte ( should be log2size - 4, since first entry is 0 ) */
int
permutation_load_2( int size, char *fipL, char *fipR, int *d_leftp,
	int *d_rightp )
{
	int ret = 0, fd, i ;
	int *ipL = NULL, *ipR = NULL ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: size %d L %s R %s lp %p rp %p\n",
		__func__, size, fipL, fipR, d_leftp, d_rightp ) ;
#endif 

	if (!( ipR = ( int * )malloc ( size ))) 
	{
		printf("%s: malloc R fail: %d\n", __func__, size  ) ;
		goto fail_out ;
	}

	if (!( ipL = ( int * )malloc ( size ))) 
	{
		printf("%s: malloc L fail: %d\n", __func__, size  ) ;
		goto fail_out ;
	}

	memset ( ipL, 0, size ) ;
	memset ( ipR, 0, size ) ;

	fd = open ( fipL, O_RDONLY ) ;
	if ( fd < 0 )
	{
		printf("%s:open L %s fail: %d\n", __func__, fipL  ) ;
		goto fail_out ;
	}

	size -= sizeof ( int ) ;	// first entry is 0

	if (( i = read( fd, ipL + 1, size ) <= 0 ) ||
		(( i != size ) && ( i % sizeof( int ))))
	{
		printf("%s:read L fail: %d\n", __func__, i ) ;
		goto fail_out ;
	}

	close( fd ) ;

	fd = open ( fipR, O_RDONLY ) ;
	if ( fd < 0 )
	{
		printf("%s:open R %s fail: %d\n", __func__, fipR  ) ;
		goto fail_out ;
	}

	if (( i = read( fd, ipR + 1, size ) <= 0 ) ||
		(( i != size ) && ( i % sizeof( int ))))
	{
		printf("%s:read R fail: %d\n", __func__, i ) ;
		goto fail_out ;
	}

	size += sizeof ( int ) ;	// include the first entry 
	
	if (( i = cudaMemcpy( d_leftp, ipL, size, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		printf("%s: download L fail: %d\n", __func__, i ) ;
		goto fail_out ;
	}

#ifdef CUDA_DBG 
	dbg_p_d_data_i("perm_tbl_left", d_leftp, ( size >> 2 )) ; 
#endif 

	if (( i = cudaMemcpy( d_rightp, ipR, size, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		printf("%s: download R fail: %d\n", __func__, i ) ;
		goto fail_out ;
	}

#ifdef CUDA_DBG 
	dbg_p_d_data_i("perm_tbl_right", d_rightp, ( size >> 2 )) ; 
#endif 

	ret = 1 ;

fail_out:

	if ( ipL )
		free ( ipL ) ;

	if ( ipR )
		free ( ipR ) ;

	if ( fd > 0 )
		close ( fd ) ;

	return ( ret ) ;
}

// first/second ... ->new

__global__ void d_do_permutation_double ( int *d_first, int *d_second,
	int *d_final, int size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < size )
	{
		d_final[ t_idx ] = d_second[ d_first[ t_idx ]] ;  

		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_permutation_double ( int *d_first, int *d_second, int *d_final,
	int tbl_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG1 
	fprintf(stderr, "%s: first %p second %p final %p tblsize %d\n",
		__func__, d_first, d_second, d_final, tbl_size ) ;
#endif 

#ifdef CUDA_DBG 
	dbg_p_d_data_i("double first", d_first, tbl_size ) ; 
	dbg_p_d_data_i("double second", d_second, tbl_size ) ; 
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_double <<< nBlocks, nThreadsPerBlock >>> (
		d_first, d_second, d_final, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("double final", d_final, tbl_size ) ; 
#endif 
}

#ifdef CUDA_OBS 
// R is the inner one to the WHM ... so it is WHM * R * RR

__global__ void d_do_permutation_RR ( int *dR, int *dRR, int *dNRR, int size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < size )
	{
		dNRR[ t_idx ] = dR[ dRR[ t_idx ]] ;  
		// dpNLL[ t_idx ] = dpLL[ dpL[ t_idx ]] ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_permutation_RR ( int *d_perm_tblR, int *d_perm_tblRR, int *d_perm_tblNRR,
	int tbl_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	fprintf(stderr, "%s: permR %p permRR %p NpermRR %p tblsize %d\n",
		__func__, d_perm_tblR, d_perm_tblRR, d_perm_tblNRR, tbl_size ) ;
#endif 

#ifdef CUDA_DBG 
	dbg_p_d_data_i("RRight perm before", d_perm_tblR, tbl_size ) ; 
	dbg_p_d_data_i("RRightRight perm before", d_perm_tblRR, tbl_size ) ; 
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_RR <<< nBlocks, nThreadsPerBlock >>> (
		d_perm_tblR, d_perm_tblRR, d_perm_tblNRR, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("New RightRight perm after", d_perm_tblNRR, tbl_size ) ; 
#endif 
}

// L is the inner one to the WHM ... so it is LL * L *  WHM

__global__ void d_do_permutation_LL ( int *dpL, int *dpLL, int *dpNLL,
	int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		dpNLL[ t_idx ] = dpLL[ dpL[ t_idx ]] ;
		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_permutation_LL ( int *d_perm_tblL, int *d_perm_tblLL,
	int *d_perm_tblNLL, int tbl_size )
{
	int nThreadsPerBlock = 512;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: permL %p permLL %p permNLL %p tblsize %d\n",
		__func__, d_perm_tblL, d_perm_tblLL, d_perm_tblNLL, tbl_size ) ;
#endif 

#ifdef CUDA_DBG 
	dbg_p_d_data_i("Left perm before", d_perm_tblL, tbl_size ) ; 
	dbg_p_d_data_i("LeftLeft perm after", d_perm_tblLL, tbl_size ) ; 
#endif 

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_permutation_LL <<< nBlocks, nThreadsPerBlock >>> (
		d_perm_tblL, d_perm_tblLL, d_perm_tblNLL, tbl_size ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("New LeftLeft perm after", d_perm_tblNLL, tbl_size ) ; 
#endif 
}
#endif 
