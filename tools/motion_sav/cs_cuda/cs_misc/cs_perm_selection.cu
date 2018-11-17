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
#include "cs_perm_selection.h"

// #define CUDA_DBG
#define CUDA_DBG1

__global__ void d_do_perm_selection_L ( int *dp, int tbl_size, 
	int *cubep, int cube_size, int random )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i ;

	while ( t_idx < cube_size )
	{
		i = cubep[ t_idx ] ;
		i = ( i + random ) % tbl_size ;

		// dp[ i ] = t_idx ;
		dp[ t_idx ] = i ;	// 

		t_idx += CUDA_MAX_THREADS ;
	}		
}

void
h_do_perm_selection_L ( int *d_perm_tbl, int tbl_size, int *d_perm_tbl_cube,
	int cube_size, int random, int sink )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; //= ( cube_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	// note: the nBlocks is based on cube_size ;

#ifdef CUDA_OBS 
	fprintf(stderr, "%s: perm %p tblsize %d cube %p cubesize %d random %d\n",
		__func__, d_perm_tbl, tbl_size, d_perm_tbl_cube, cube_size,
		random ) ;
#endif 

	set_device_mem_i ( d_perm_tbl, tbl_size, ( sink + random ) % tbl_size ) ;
	
	h_block_adj ( cube_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_perm_selection_L <<< nBlocks, nThreadsPerBlock >>> (
		d_perm_tbl, tbl_size, d_perm_tbl_cube, cube_size, random ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_DBG 
	dbg_p_d_data_i("h_do_perm_selection_L", d_perm_tbl, tbl_size ) ; 
#endif 
}

__global__ void d_do_perm_selection_R ( int *dp, int tbl_size, int random )
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i ;

	while ( t_idx < tbl_size )
	{
		if ( t_idx == 0 )
			dp[ t_idx ] = 0 ;
		else
		{
			i = t_idx + random ;
			dp[ t_idx ] = i % tbl_size ;	
			if ( i / tbl_size )
				dp[ t_idx ]++ ;		// take out 0
		}
		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_do_perm_selection_R ( int *d_perm_tbl, int tbl_size, int random )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; //= ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: perm %p size %d random %d\n",
		__func__, d_perm_tbl, tbl_size, random ) ;
#endif 

#ifdef CUDA_DBG 
	if ( tbl_size <= random )
		fprintf( stderr, "%s: ERROR tblsize %d >= random %d\n",
			__func__, tbl_size, random ) ;
#endif 
	
	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_do_perm_selection_R <<< nBlocks, nThreadsPerBlock >>> (
		d_perm_tbl, tbl_size, random ) ; 

	cudaThreadSynchronize() ;

#ifdef CUDA_OBS 
	dbg_p_d_data_i("h_do_perm_selection_R", d_perm_tbl, tbl_size ) ; 
#endif 
}

/*
h_do_get_perm_matrix:
   the purpose of this routine is to mark the location, i.e. index, of the elements inside
   the cube in relation to the location inside the block 
*/
void
h_do_get_perm_matrix( int *dp, int ox, int oy, int oz,
	int cx, int cy, int cz, int *sinkp )
{
	int sink = -1, idx, i, j, k, frame_size ;

	frame_size = ox * oy ;

	for ( i = 0 ; i < cz ; i++ )
	{
		idx = i * frame_size ;
		for ( j = 0 ; j < cy ; j++ )
		{
			for ( k = 0 ; k < cx ; k++ )
				*dp++ = idx++ ;

			if (( sink < 0 ) && ox != cx )
				sink = cx ;

			idx += ( ox - cx ) ;
		}

		if (( sink < 0 ) && oy != cy )
			sink = cy * ox  ;
	}
	if ( sink < 0 ) 
	{
		if ( oz != cz )
			sink = frame_size * cz ;
		else
			sink = 0 ; // will be over-written anyway, so just give a number
	}
	*sinkp = sink ;
}

// try to fit shifting cube in block ...

int
cube_size( int *p )
{
	int k, i ;

	k = 1 ;
	i = 3 ;
	while ( i-- )
		k *= *p++ ;

	return ( k ) ;
}

void
ck_blk( char *s, int *small, int *large  )
{
	int i ;

	i = 3 ;
	while ( i-- )
	{
		if ( small[i] > large[i] )
		{
			printf("%s: %s small %d %d %d large %d %d %d\n",
				__func__, s,
				small[0], small[1], small[2],
				large[0], large[1], large[2]) ;
			exit( 33 ) ;	
		}
	}
}

void
ck_size( const char *s, int *p, int size )
{
	int k, i ;

	k = 1 ;
	i = 3 ;
	while ( i-- )
		k *= *p++ ;

	if ( k > size )
	{
		printf("%s: %s got %d need %d\n", __func__, s, k, size ) ;
		exit( 33 ) ;	
	}
}

int
h_do_find_perm_size( int bx, int by, int bz, int *cx,
	int *cy, int *cz, int max_z, int nmea, int min_x, int min_y )
{
	double f ;
	int dox, done_once, bb[3], cc[3], xy, yz, xz, i, j, k ;

	bb[0] = bx ;	// block
	bb[1] = by ;
	bb[2] = bz ;

#ifdef CUDA_DBG1 
	printf("%s: block %d %d %d cube %d %d %d max_z %d nmea %d min x/y %d %d\n",
		__func__, bx, by, bz, *cx, *cy, *cz, max_z, nmea, min_x, min_y ) ;
#endif 

	k = cube_size( bb ) ;

	if ( nmea >= k )
	{
		*cx = bx ;
		*cy = by ;
		*cz = bz ;

		return ( 1 ) ;
	}

	cc[0] = *cx ;     // selection
	cc[1] = *cy ;
	cc[2] = *cz ;

	if (( cc[0] > bb[0] ) || ( cc[1] > bb[1] ) || ( cc[2] > bb[2] ))
	{
#ifdef CUDA_DBG1 
		printf("size mismatch: %d %d %d -- %d %d %d -- %d\n", cc[0], cc[1], cc[2],
			 bb[0], bb[1], bb[2], nmea ) ;
#endif 

		return ( 0 ) ;
	}


#ifdef CUDA_DBG1 
	printf("%s: init: %d %d %d -- %d %d %d -- %d\n", __func__, cc[0], cc[1],
		cc[2], bb[0], bb[1], bb[2], nmea ) ;
#endif 

	i = cube_size( cc ) ;
	if ( !i )
	{
#ifdef CUDA_DBG1
		printf("%s: size 0: %d %d %d -- %d %d %d -- %d\n", __func__,
			cc[0], cc[1], cc[2], bb[0], bb[1], bb[2], nmea ) ;
#endif 

		return ( 0 ) ;
	}

	f = ( double )nmea / ( double )i ;

#ifdef CUDA_DBG1
	printf("2: f %f i %d \n", f, i ) ;
#endif 

	if ( f < 1.0 )	// razi ... 
	{
#ifdef CUDA_DBG1 
		printf("%s:less than 1.0: %d %d %d -- %d %d %d -- %d f %f\n",
			__func__, cc[0], cc[1], cc[2], bb[0], bb[1], bb[2], nmea, f ) ;
#endif 
		return ( 0 ) ;
	}

	f = pow ( f, 1.0/3.0 ) ;

	// it will not shrink ... razi

	i = 3 ;
	while ( i-- )
		cc[i] = ( int )( f * ( double ) cc[i] ) ;

	if ( cc[2] > max_z ) 
		cc[2] = max_z ;

#ifdef CUDA_DBG1 
	printf("%s: max: %d %d %d t %d -- %f mea %d \n", __func__,
		cc[0], cc[1], cc[2], cube_size( cc ), f, nmea ) ;
#endif 

#ifdef CUDA_DBG1 
	ck_size( "first adjust", cc, nmea ) ;
#endif 

	// ok ... less than nmeas ... make sure it is inside the blk

	i = 3 ;
	while ( i-- )
	{
		if ( cc[i] > bb[i] )
		{
			f = (( double ) bb[i] ) / (( double ) cc[i] ) ;
			for ( j = 0 ; j < 3 ; j++ )
				cc[j] = ( int )(( double )cc[j] * f + 0.5 ) ;	// round up
		}
	}

	if ( cc[2] > max_z ) 
		cc[2] = max_z ;

	if ( cc[0] < min_x )
		cc[0] = min_x ;

	if ( cc[1] < min_y )
		cc[1] = min_y ;

	i = nmea / ( cc[0] * cc[1] ) ;

	if ( cc[2] > i )
		cc[2] = i ; 

#ifdef CUDA_DBG1
	ck_size( "inside the box", cc, nmea ) ;
#endif 

	// ok ... less than nmeas
	// ok ... inside the block

#ifdef CUDA_DBG1
	printf("%s: inside the box: %d %d %d t %d -- %f -- max %d\n", __func__,
		cc[0], cc[1], cc[2], cc[0]* cc[1]* cc[2], f, max_z ) ;
#endif 

	// ok ... now increase the size ...

	done_once = 0 ;

	dox = 1 ;
	while ( 1 )
	{
		xy = cc[0] * cc[1] ;
		xz = cc[0] * cc[2] ;
		yz = cc[1] * cc[2] ;

		k = nmea - cube_size( cc ) ;

		done_once++ ;
		if (( cc[0] > min_x ) && ( cc[1] > min_y ) && ( k >= xy ) && ( cc[2] < bz ) && ( cc[2] < max_z ))
		{
		   cc[2]++ ;
		   done_once = 0 ;
		} else 
		{
			if ( dox )
			{
				dox = 0 ;
				if (( k >= yz ) && ( cc[0] < bx ))
				{
					done_once = 0 ;
					cc[0]++ ;
				} else if (( k >= xz ) && ( cc[1] < by ))
				{
					cc[1]++ ;
					done_once = 0 ;
				} 
			} else
			{
				dox = 1 ;
				if (( k >= xz ) && ( cc[1] < by ))
				{
					cc[1]++ ;
					done_once = 0 ;
				} else if (( k >= yz ) && ( cc[0] < bx ))
				{
					cc[0]++ ;
					done_once = 0 ;
				}
			}
		}	

#ifdef CUDA_DBG1 
		printf("%s: searching: %d %d %d t %d -- done %d\n", __func__,
			cc[0], cc[1], cc[2], cube_size( cc ), done_once ) ;
#endif 

		if ( done_once == 3 )
			break ;
	}

#ifdef CUDA_OBS1 
	printf("%s: winner: %d %d %d t %d %d -- %d\n", __func__,
		cc[0], cc[1], cc[2], cube_size( cc ), nmea, nmea - cube_size(cc) ) ;
#endif 

	*cx = cc[0] ;
	*cy = cc[1] ;
	*cz = cc[2] ;

	return ( 1 ) ;
}
