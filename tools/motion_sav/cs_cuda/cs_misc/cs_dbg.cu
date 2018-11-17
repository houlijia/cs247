#include <stdio.h>
#include <errno.h>
#include "cs_header.h"
#include "cs_dbg.h"

#define CUDA_DBG 

#define DBG_BUF_SIZE (1024 * 1024)

int *dbg_bufp, dbg_size ;

static char dbg_msg[ 200 ] ;

void dbg_pdata_ll( const char *s, long long *dp, int size ) ;

int
dbg_init( int size )
{
	if (!( dbg_bufp = ( int * ) malloc ( size )))
	{
	  fprintf( stderr, "dbg_init: malloc failed(%d): %s \n", errno, strerror(errno) ) ;
		return ( 0 ) ;
	}
	dbg_size = size ;
	return ( 1 ) ;
}

void
dbg_clear_buf( int *cp, int size )
{
	memset ( cp, 0, size ) ;
}

void
dbg_set_buf( int *cp, int size, int set )
{
	while ( size-- )
		*cp++ = set++ ; 
}

void
dbg_p_d_data_ll ( const char *s, long long *dp, int size )
{
	fprintf( stderr, "%s: %s size %d dp %p\n", 
		__func__, s, size, dp ) ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p\n", 
		__func__, s, size, dp ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size /= sizeof ( long long ) ;

	dbg_pdata_ll ( s, ( long long *)dbg_bufp, size ) ; 
}

void
dbg_p_data_i_mn ( const char *s, int *dp, int size, int m, int n, int doprint )
{
	int *otp, *tp, i, j ;

	fprintf( stderr, "%s: %s size %d m %d n %d doprint %d dp %p\n", 
		__func__, s, size, m, n, doprint, dp ) ;

	size /= ( m * n ) ;

	otp = dp ;
	while ( size-- )
	{
		for ( i = 0 ; i < n ; i++ )
		{
			tp = otp ;
			for ( j = 0 ; j < doprint ; j++ )
				printf("%d	", *tp++ ) ;
			printf("\n") ;
			otp += m ;
		}
		printf("\n") ;
	}
}

void
dbg_p_data_md_f_mn ( const char *s, int *dp, int size, int m, int n, int doprint )
{
	int *otp, *tp, i, j ;
	float *fp ;

	fprintf( stderr, "%s: %s size %d m %d n %d doprint %d dp %p\n", 
		__func__, s, size, m, n, doprint, dp ) ;

	size /= ( m * n ) ;

	otp = dp ;
	while ( size-- )
	{
		for ( i = 0 ; i < n ; i++ )
		{
			tp = otp ;
			for ( j = 0 ; j < doprint ; j++ )
			{
				if (!(( j + 1 ) % 4 ))
				{
					fp = ( float *)tp++ ;
					printf("%f	", *fp ) ;
				} else
					printf("%d	", *tp++ ) ;
			}
			printf("\n") ;
			otp += m ;
		}
		printf("\n") ;
	}
}

void
dbg_p_d_data_c_mn ( const char *s, char *dp, int size, int m, int n, int doprint )
{
	char *otp, *tp ;
	int i, j ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d m %d n %d dp %p\n", 
		__func__, s, size, m, n, dp ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size /= ( m * n ) ;

	otp = ( char *)dbg_bufp ;
	while ( size-- )
	{
		for ( i = 0 ; i < n ; i++ )
		{
			tp = otp ;
			for ( j = 0 ; j < doprint ; j++ )
				printf("%d	", *tp++ ) ;
			printf("\n") ;
			otp += m ;
		}
		printf("\n") ;
	}
}

void
dbg_p_d_data_i_mn_skip ( const char *s, int *dp, int size, int m, int n, int z, int doprint, int perm_size )
{
	int ii, *otp, *fp, *tp, i, j, k ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s m %d n %d size %d dp %p perm_size %d\n", 
		__func__, s, m, n, size, dp, perm_size ) ;

	if (( m * n * z ) > perm_size )
	{
		fprintf( stderr, "%s: err m %d n %d z %d > perm %d \n",
			__func__, m, n, z, perm_size ) ;
		return ;
	} 

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	size /= perm_size ;

	fp = dbg_bufp ;
	for ( ii = 0 ; ii < size ; ii++ )
	{
		printf("perm === %d\n", ii ) ;
		otp = fp ;
		for ( k = 0 ; k < z ; k++ )
		{
			printf("perm %d z %d \n", ii, k ) ;
			for ( i = 0 ; i < n ; i++ )
			{
				printf("z %d y %d\n", k, i ) ;
				tp = otp ;
				for ( j = 0 ; j < doprint ; j++ )
					printf("%d	", *tp++ ) ;
				printf("\n") ;
				otp += m ;
			}
			printf("\n") ;
		}

		fp += perm_size ;
	}
}

void
dbg_p_d_data_f_mn_skip ( const char *s, float *dp, int size, int m, int n, int z, int doprint, int perm_size )
{
	int ii, i, j, k ;
	float *fp, *tp, *otp ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s m %d n %d size %d dp %p perm_size %d\n", 
		__func__, s, m, n, size, dp, perm_size ) ;

	if (( m * n * z ) > perm_size )
	{
		fprintf( stderr, "%s: err m %d n %d z %d > perm %d \n",
			__func__, m, n, z, perm_size ) ;
		return ;
	} 

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	size /= perm_size ;

	fp = ( float *)dbg_bufp ;
	for ( ii = 0 ; ii < size ; ii++ )
	{
		printf("perm === %d\n", ii ) ;
		otp = fp ;
		for ( k = 0 ; k < z ; k++ )
		{
			printf("perm %d z %d \n", ii, k ) ;
			for ( i = 0 ; i < n ; i++ )
			{
				printf("z %d y %d\n", k, i ) ;
				tp = otp ;
				for ( j = 0 ; j < doprint ; j++ )
					printf("%.4f	", *tp++ ) ;
				printf("\n") ;
				otp += m ;
			}
			printf("\n") ;
		}

		fp += perm_size ;
	}
}

void
dbg_p_d_data_f_mn ( const char *s, float *dp, int size, int m, int n, int doprint )
{
	float *otp, *tp ;
	int i, j ;

	fprintf( stderr, "%s: %s m %d n %d size %d dp %p dbgsize %d\n", 
		__func__, s, m, n, size, dp, dbg_size ) ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s m %d n %d size %d dp %p dbgsize %d\n", 
		__func__, s, m, n, size, dp, dbg_size ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	size /= ( m * n ) ;

	otp = ( float *)dbg_bufp ;
	while ( size-- )
	{
		for ( i = 0 ; i < n ; i++ )
		{
			tp = otp ;
			for ( j = 0 ; j < doprint ; j++ )
				printf("%.2f	", *tp++ ) ;
			printf("\n") ;
			otp += m ;
		}
		printf("\n") ;
	}
}

void
dbg_p_d_data_f_mn ( const char *s, float *dp, int size, int m, int n, int doprint, int printrow )
{
	float *otp, *tp ;
	int i, j, k ;

	fprintf( stderr, "%s: %s m %d n %d size %d dp %p dbgsize %d printrow %d\n", 
		__func__, s, m, n, size, dp, dbg_size, printrow ) ;

	if ( printrow > n )
		printrow = n ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s m %d n %d size %d dp %p dbgsize %d\n", 
		__func__, s, m, n, size, dp, dbg_size ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	size /= ( m * n ) ;

	fprintf( stderr, "%s: size %d \n", __func__, size ) ;

	otp = ( float *)dbg_bufp ;

	for ( k = 0 ; k < size ; k++ )
	{
		printf("\n%s BLK ::: %d \n", __func__, k ) ;

		for ( i = 0 ; i < printrow ; i++ )
		{
			tp = otp ;
			for ( j = 0 ; j < doprint ; j++ )
				printf("%.2f	", *tp++ ) ;
			printf("\n") ;
			otp += m ;
		}

		otp += ( n - printrow ) * m ;
		printf("\n") ;
	}
}

void
dbg_p_d_data_i_mn ( const char *s, int *dp, int size, int m, int n, int doprint )
{
	int *otp, *tp, i, j ;

	fprintf( stderr, "%s: %s m %d n %d size %d dp %p dbgsize %d\n", 
		__func__, s, m, n, size, dp, dbg_size ) ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s m %d n %d size %d dp %p dbgsize %d\n", 
		__func__, s, m, n, size, dp, dbg_size ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	size /= ( m * n ) ;

	otp = dbg_bufp ;
	while ( size-- )
	{
		for ( i = 0 ; i < n ; i++ )
		{
			tp = otp ;
			for ( j = 0 ; j < doprint ; j++ )
				printf("%d	", *tp++ ) ;
			printf("\n") ;
			otp += m ;
		}
		printf("\n") ;
	}
}

void
dbg_p_data_i_mn_v2 ( const char *s, int *hp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y )
{
	int ddoprint, tt, t, ii, k, xyz_size, idx, m, n, *btp, *otp, *tp, i, j ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p blk x/y %d %d\n", 
		__func__, s, size, hp, blk_in_x, blk_in_y  ) ;

	dbg_get_d_data (( char *)hp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	xyz_size = dp[0].x * dp[0].y * dp[0].z ;

	size /= xyz_size ;

	btp = dbg_bufp ;

	printf("%s: size %d xyz %d \n", __func__, size, xyz_size ) ;

	while ( 1 ) 
	{
		for ( j = 0 ; j < blk_in_y ; j++ ) 
		{
			for ( i = 0 ; i < blk_in_x ; i++ )
			{
				otp = btp ;
				if (( i == 0 ) || ( i == ( blk_in_x - 1 )))
				{
					if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
						idx = 2 ;
					else
						idx = 1 ;
				} else
				{
					if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
						idx = 1 ;
					else
						idx = 0 ;
				}

				m = dp[idx].x ;
				n = dp[idx].y ;
				t = dp[idx].z ;

				printf("%s: i %d j %d m/n/t %d %d %d \n",
					__func__, i, j, m, n, t ) ;

				ddoprint = ( doprint > m ) ? m : doprint ;

				for ( tt = 0 ; tt < t ; tt++ )
				{ 
					for ( ii = 0 ; ii < n ; ii++ )
					{
						tp = otp ;
						for ( k = 0 ; k < ddoprint ; k++ )
							printf("%d	", *tp++ ) ;
						printf("\n") ;
						otp += m ;
					}
					printf("\n") ;
				}

				printf("\n") ;
				btp += xyz_size ;

				if ( --size == 0 )
					return ;
			}
		}
	}
}

// this is exactly the same as dbg_p_d_data_i_mn_v2 ... other than the printf(). template?
void
dbg_p_d_data_f_mn_v2 ( const char *s, float *devp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y )
{
	int ddoprint, tt, t, ii, k, xyz_size, idx, m, n, i, j ;
	float *btp, *otp, *tp ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p blk x/y %d %d\n", 
		__func__, s, size, devp, blk_in_x, blk_in_y  ) ;

	dbg_get_d_data (( char *)devp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	xyz_size = dp[0].x * dp[0].y * dp[0].z ;

	size /= xyz_size ;

	btp = ( float *)dbg_bufp ;

	printf("%s: size %d xyz %d \n", __func__, size, xyz_size ) ;

	while ( 1 ) 
	{
		for ( j = 0 ; j < blk_in_y ; j++ ) 
		{
			for ( i = 0 ; i < blk_in_x ; i++ )
			{
				otp = btp ;
				if (( i == 0 ) || ( i == ( blk_in_x - 1 )))
				{
					if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
						idx = 2 ;
					else
						idx = 1 ;
				} else
				{
					if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
						idx = 1 ;
					else
						idx = 0 ;
				}

				m = dp[idx].x ;
				n = dp[idx].y ;
				t = dp[idx].z ;

				printf("%s: x %d y %d m/n/t %d %d %d otp %p\n",
					__func__, i, j, m, n, t, otp ) ;

				ddoprint = ( doprint > m ) ? m : doprint ;

				for ( tt = 0 ; tt < t ; tt++ )
				{ 
					for ( ii = 0 ; ii < n ; ii++ )
					{
						tp = otp ;
						for ( k = 0 ; k < ddoprint ; k++ )
							printf("%.4f	", *tp++ ) ;
						printf("\n") ;
						otp += m ;
					}
					printf("\n") ;
				}

				printf("\n") ;
				btp += xyz_size ;

				if ( --size == 0 )
					return ;
			}
		}
	}
}

void
dbg_p_d_data_i_mn_v2 ( const char *s, int *devp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y )
{
	int ddoprint, tt, t, ii, k, xyz_size, idx, m, n, *btp, *otp, *tp, i, j ;

	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p blk x/y %d %d\n", 
		__func__, s, size, devp, blk_in_x, blk_in_y  ) ;

	dbg_get_d_data (( char *)devp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	xyz_size = dp[0].x * dp[0].y * dp[0].z ;

	size /= xyz_size ;

	btp = dbg_bufp ;

	printf("%s: size %d xyz %d \n", __func__, size, xyz_size ) ;

	while ( 1 ) 
	{
		for ( j = 0 ; j < blk_in_y ; j++ ) 
		{
			for ( i = 0 ; i < blk_in_x ; i++ )
			{
				otp = btp ;
				if (( i == 0 ) || ( i == ( blk_in_x - 1 )))
				{
					if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
						idx = 2 ;
					else
						idx = 1 ;
				} else
				{
					if (( j == 0 ) || ( j == ( blk_in_y - 1 )))
						idx = 1 ;
					else
						idx = 0 ;
				}

				m = dp[idx].x ;
				n = dp[idx].y ;
				t = dp[idx].z ;

				printf("%s: i %d j %d m/n/t %d %d %d otp %p\n",
					__func__, i, j, m, n, t, otp ) ;

				ddoprint = ( doprint > m ) ? m : doprint ;

				for ( tt = 0 ; tt < t ; tt++ )
				{ 
					for ( ii = 0 ; ii < n ; ii++ )
					{
						tp = otp ;
						for ( k = 0 ; k < ddoprint ; k++ )
							printf("%d	", *tp++ ) ;
						printf("\n") ;
						otp += m ;
					}
					printf("\n") ;
				}

				printf("\n") ;
				btp += xyz_size ;

				if ( --size == 0 )
					return ;
			}
		}
	}
}

/* print double content from device ... size is in char */
void
dbg_p_d_data_d ( const char *s, float *dp, int size )
{
	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p\n", 
		__func__, s, size, dp ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	dbg_pdata_d ( s, ( float *)dbg_bufp, size ) ; 
}

void
dbg_p_d_data_i ( const char *s, int *dp, int size )
{
	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p\n", 
		__func__, s, size, dp ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	dbg_pdata_i ( s, ( int *)dbg_bufp, size ) ; 
}

void
dbg_p_d_data_f ( const char *s, float *dp, int size )
{
	size <<= 2 ;	// work with float and int

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p\n", 
		__func__, s, size, dp ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	dbg_pdata_f ( s, ( float *)dbg_bufp, size ) ; 
}

void
dbg_p_d_data_c ( const char *s, char *dp, int size )
{
	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "%s: %s size %d dp %p\n", 
		__func__, s, size, dp ) ;

	dbg_get_d_data ( dp, ( char *)dbg_bufp, size ) ;

	dbg_pdata_c ( s, ( char *)dbg_bufp, size ) ; 
}

int
dbg_copy_d_data ( char *dtp, char *dfp, int size ) 
{
	int i ;

	if (( i = cudaMemcpy( dtp, dfp, size, cudaMemcpyDeviceToDevice)) !=
		cudaSuccess )
	{
		fprintf( stderr, "dbg_copy_d_data: failed %d\n", i ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

int
dbg_put_d_data ( char *dp, char *hp, int size ) 
{
	int i ;

	if (( i = cudaMemcpy( dp, hp, size, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		fprintf( stderr, "dbg_put_d_data: failed %d\n", i ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

int
dbg_get_d_data ( char *dp, char *hp, int size ) 
{
	int i ;

	if (( i = cudaMemcpy( hp, dp, size, cudaMemcpyDeviceToHost)) !=
		cudaSuccess )
	{
		fprintf(stderr, "dbg_get_d_data: hp %p dp %p size %d failed %d\n", hp, dp, size, i ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

void
dbg_pdata_ll( const char *s, long long *dp, int size )
{
	int i ; 

	fprintf( stderr, "dbg_pdata_ll: %s\n", s ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %p -- %lld 0x%llx\n", i, dp, *dp, *dp ) ;
		i++ ;
		dp++ ;
	}
}

void
dbg_pdata_d( const char *s, const float *dp, int size )
{
	int i ;

	fprintf( stderr, "%s: %s\n", __func__, s ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %p == %f\n", i, dp, *dp ) ;
		i++ ;
		dp++ ;
	}
}

void
dbg_pdata_i( const char *s, const int *dp, int size )
{
	int i ;

	fprintf( stderr, "%s: %s\n", __func__, s ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %8.8x %d\n", i, *dp, *dp ) ;
		i++ ;
		dp++ ;
	}
}

void
dbg_pdata_f( const char *s, const float *dp, int size )
{
	int i ;

	fprintf( stderr, "%s: %s dp %p size %d\n", __func__, s, dp, size ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %f\n", i, *dp ) ;
		i++ ;
		dp++ ;
	}
}

void
dbg_pdata_c( const char *s, const char *dp, int size )
{
	int i ;
	unsigned char *cp = ( unsigned char *)dp ;

	fprintf( stderr, "%s: %s\n", __func__, s ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %2.2x %d\n", i, *cp, *cp) ;
		i++ ;
		cp++ ;
	}
}

void
dbg_mdata( int *dp, int size )
{
	int cnt, k, i ;

	cnt = 0 ;
	k = 0 ;
	while ( size-- )
	{
		if ( k != EOF )
		k = scanf("%d", &i ) ;

		if ( k == EOF )
			*dp++ = 0 ;
		else
		{
			cnt++ ;
			*dp++ = i ;
		}
	}
	// printf("makedata: data cnt %d\n", cnt ) ;
}    

float *
dbg_d_malloc_f ( int size )
{
	float *cp ;
	int i ;

	if (( i = cudaMalloc( &cp, size * sizeof( float ))) != cudaSuccess )
	{
		printf("%s: 2 cudaMalloc failed %d\n", __func__, i ) ;
		return ( NULL ) ;
	}
	return ( cp ) ;
}

int *
dbg_d_malloc_i ( int size )
{
	int *cp ;
	int i ;

	if (( i = cudaMalloc( &cp, size * sizeof( int ))) != cudaSuccess )
	{
		printf("%s: 2 cudaMalloc failed %d\n", __func__, i ) ;
		return ( NULL ) ;
	}
	return ( cp ) ;
}

char *
dbg_d_malloc_c ( int size )
{
	char *cp ;
	int i ;

	if (( i = cudaMalloc( &cp, size )) != cudaSuccess )
	{
		printf("%s: 2 cudaMalloc failed %d\n", __func__, i ) ;
		return ( NULL ) ;
	}
	return ( cp ) ;
}

void
dbg_p_data_i_cube ( const char *s, int *dp, int vx, int hy, int tz )
{
	int i, j, k, l ;

	fprintf( stderr, "%s: %s x/y/t %d %d %d\n", __func__, s, vx, hy, tz ) ;
	for ( i = 0 ; i < tz ; i++ )
	{
		printf("T = %d \n", i ) ;
		for ( j = 0 ; j < vx ; j++ )
		{
			printf("T %d, X %d \n", i, j ) ;
			for ( k = 0 ; k < hy ; k++ )
			{		
				for ( l = 0 ; (( l < 8 ) && ( k < hy )) ; l++ )
				{
					printf("%8.8x ", *dp++ ) ;
					k++ ;
				}

				if ( k < hy )
					k-- ;

				printf("\n") ;

				// fprintf( stderr, "%d -- %p == %f\n", i, dp, *dp ) ;
			}
			printf("\n") ;
		}
		printf("\n") ;
	}
}

// size in sizeof( int )
void
dbg_p_d_data_i_cube ( const char *s, int *devp, int vx, int hy, int tz )
{
	int size = vx * hy * tz ;

	size *= sizeof( int ) ;
	dbg_get_d_data (( char *)devp, ( char *)dbg_bufp, size ) ;

	dbg_p_data_i_cube ( s, dbg_bufp, vx, hy, tz ) ;

}

void
dbg_p_data_i_cube ( const char *s, float *dp, int vx, int hy, int tz )
{
	int i, j, k, l ;

	fprintf( stderr, "%s float: %s x/y/t %d %d %d\n", __func__, s, vx, hy, tz ) ;
	for ( i = 0 ; i < tz ; i++ )
	{
		printf("T = %d \n", i ) ;
		for ( j = 0 ; j < vx ; j++ )
		{
			printf("T %d, X %d \n", i, j ) ;
			for ( k = 0 ; k < hy ; k++ )
			{		
				for ( l = 0 ; (( l < 8 ) && ( k < hy )) ; l++ )
				{
					printf("%8.4f ", *dp++ ) ;
					k++ ;
				}

				if ( k < hy )
					k-- ;

				printf("\n") ;

				// fprintf( stderr, "%d -- %p == %f\n", i, dp, *dp ) ;
			}
			printf("\n") ;
		}
		printf("\n") ;
	}
}

// size in sizeof( float )
void
dbg_p_d_data_i_cube ( const char *s, float *devp, int vx, int hy, int tz )
{
	int size = vx * hy * tz ;

	size *= sizeof( float ) ;
	dbg_get_d_data (( char *)devp, ( char *)dbg_bufp, size ) ;

	dbg_p_data_i_cube ( s, ( float *)dbg_bufp, vx, hy, tz ) ;

}

// test of this func is done in test/cs_random_test
int
dbg_ck_unique ( char *s, int *dp, int size )
{
	int err, k, i, *oip, *ip, *dip ;

	i = ( size << 2 ) ;
	i <<= 1 ; // split the dbg_size

	if ( i > dbg_size )
	{
	  fprintf( stderr, "%s: err size %d > dbg_size %d \n", s, i, dbg_size ) ;
		return ( 0 ) ;
	}

	oip = ip = dbg_bufp ;
	dip = ip + size ;

	dbg_get_d_data (( char * )dp, ( char * ) dip, size * sizeof ( int )) ;

	i = size ;
	while ( i-- )
		*ip++ = 0 ;

	err = 0 ;
	ip = oip ;
	i = size ;
	while ( i-- )
		ip[ *dip++ ]++ ;

	k = 0 ;
	for ( i = 0 ; i < size ; i++ )
	{
		if ( *ip != 1 )
		{
			err++ ;
			fprintf( stderr, "%s : %s : idx %d val %d \n", __func__, s, i, *ip ) ;
		}

		k += *ip++ ;
	}

	if ( k != size ) 
		fprintf( stderr, "%s : %s k %d size %d \n", __func__, s, k, size ) ;

	if ( err )
		fprintf( stderr, "%s : %s err %d\n", __func__, s, err ) ;
	else
		fprintf( stderr, "%s : %s good\n", __func__, s ) ;

	return ( !err ) ;
}

void
dbg_pr_h_first_last ( char *s, char *h_p, int len, int pr_size )
{
	int i ;

	i = len >> 1 ;

	if ( pr_size > i )
		pr_size = i ;

	sprintf( dbg_msg, "%s : FIRST %d OF %d", s, pr_size, len ) ;
	dbg_pdata_c ( dbg_msg, h_p, pr_size ) ;

	sprintf( dbg_msg, "%s : CENTER %d AFTER %d", s, pr_size, i ) ;
	dbg_pdata_c ( dbg_msg, h_p + i, pr_size ) ;

	sprintf( dbg_msg, "%s : LAST %d AFTER %d", s, pr_size, len - pr_size ) ;
	dbg_pdata_c ( dbg_msg, h_p + ( len - pr_size ), pr_size ) ;
}

void
dbg_pr_h_first_last ( char *s, int *h_p, int len, int pr_size )
{
	int i ;

	i = len >> 1 ;

	if ( pr_size > i )
		pr_size = i ;

	sprintf( dbg_msg, "%s : FIRST %d OF %d", s, pr_size, len ) ;
	dbg_pdata_i ( dbg_msg, h_p, pr_size ) ;

	sprintf( dbg_msg, "%s : CENTER %d AFTER %d", s, pr_size, i ) ;
	dbg_pdata_i ( dbg_msg, h_p + i, pr_size ) ;

	sprintf( dbg_msg, "%s : LAST %d AFTER %d", s, pr_size, len - pr_size ) ;
	dbg_pdata_i ( dbg_msg, h_p + ( len - pr_size ), pr_size ) ;
}

void
dbg_pr_first_last ( char *s, int *d_p, int len, int pr_size )
{
	int i ;

	i = len >> 1 ;

	if ( pr_size > i )
		pr_size = i ;

	sprintf( dbg_msg, "%s : FIRST %d OF %d", s, pr_size, len ) ;
	dbg_p_d_data_i ( dbg_msg, d_p, pr_size ) ;

	sprintf( dbg_msg, "%s : CENTER %d AFTER %d", s, pr_size, i ) ;
	dbg_p_d_data_i ( dbg_msg, d_p + i, pr_size ) ;

	sprintf( dbg_msg, "%s : LAST %d AFTER %d", s, pr_size, len - pr_size ) ;
	dbg_p_d_data_i ( dbg_msg, d_p + ( len - pr_size ), pr_size ) ;
}

void
dbg_pr_first_last ( char *s, float *d_p, int len, int pr_size )
{
	int i ;

	i = len >> 1 ;

	if ( pr_size > i )
		pr_size = i ;

	sprintf( dbg_msg, "%s : FIRST %d OF %d", s, pr_size, len ) ;
	dbg_p_d_data_f ( dbg_msg, d_p, pr_size ) ;

	sprintf( dbg_msg, "%s : CENTER %d AFTER %d", s, pr_size, i ) ;
	dbg_p_d_data_f ( dbg_msg, d_p + i, pr_size ) ;

	sprintf( dbg_msg, "%s : LAST %d AFTER %d", s, pr_size, len - pr_size ) ;
	dbg_p_d_data_f ( dbg_msg, d_p + ( len - pr_size ), pr_size ) ;
}

// size in number of int
int
dbg_perm_ck ( int callid, int callid2, int zero_in_zero, int *d_bp, int size )
{
	int bsize, *oip, i, j, *fip ;

	fprintf( stderr, "%s: call %d id %d zero %d bp %p size %d\n", 
		__func__, callid, callid2, zero_in_zero, d_bp, size ) ;

	bsize = size << 2 ;

	if ( bsize > dbg_size )
	{
		fprintf( stderr, "dbg_perm_ck: err size %d too big  dbg_size %d \n", size, dbg_size ) ;
		return ( 0 ) ;
	}

	dbg_get_d_data (( char *)d_bp, ( char *)dbg_bufp, bsize ) ;

	oip = dbg_bufp + size ;

	memset ( oip, 0, bsize ) ;

	fip = dbg_bufp ;

	if ( zero_in_zero )
	{
		if ( *fip )
		{
			fprintf( stderr, "%s: err zero : %d \n",
				__func__, *fip ) ;
			dbg_pdata_i ( "dbg_perm_ck", dbg_bufp, size ) ;
			return ( 0 ) ;
		}

		fip++ ;
		oip[0] = 1 ;
		i = 1 ;
	} else
		i = 0 ;

	for ( ; i < size ; i++ )
	{
		j = *fip++ ;

		if (( j < 0 ) || ( j > size ))
		{
			fprintf( stderr, "%s: err too big : idx %d entry %d double \n",
				__func__, i, j ) ;
			dbg_pdata_i ( "dbg_perm_ck", dbg_bufp, size ) ;
		    return ( 0 ) ;
		}

		if ( oip[j] )
		{	
			fprintf( stderr, "%s: err idx %d entry %d double \n",
				__func__, i, j ) ;
			dbg_pdata_i ( "dbg_perm_ck", dbg_bufp, size ) ;
		    return ( 0 ) ;
		}

		oip[j]++ ;
	}

	for ( i = 0 ; i < size ; i++ )
	{
		if ( !oip[i] ) 
		{	
			fprintf( stderr, "%s: err missing idx %d\n",
				__func__, i ) ;
			dbg_pdata_i ( "dbg_perm_ck", dbg_bufp, size ) ;
		    return ( 0 ) ;
		}
	}

	return ( 1 ) ;
}

// record_size does not include the NUM_OF_HVT_INDEX
void
cs_p_d_tvh ( const char *s, int *dmem, int record_size, int num_rec, int do_print, int do_skip )
{
	int do_print2, i, j, *ip, *oip ; 
	float *fp ;

	printf("%s ::: dmem %p record_size %d num_rec %d doprint %d do_skip %d\n",
		s, dmem, record_size, num_rec, do_print, do_skip ) ;

	dbg_get_d_data (( char *)dmem, ( char *)dbg_bufp,
		sizeof( int ) * ( record_size + 3 ) * num_rec ) ;

	if ( do_print > record_size )
		do_print = record_size ;

	if ( do_skip > record_size )
		do_skip = 0 ;

	if ( do_skip )
	{
		do_print2 = (( record_size - do_skip ) > do_print ) ?  do_print : ( record_size - do_skip ) ;
	}

	oip = dbg_bufp ;
	for ( i = 0 ; i < num_rec ; i++ )
	{
		ip = oip ;
		printf("IDX record %d t %d v %d h %d ", i, ip[0], ip[1], ip[2] ) ;

		fp = ( float *)oip + 3 ; // is NUM_OF_HVT_INDEX
		printf(" --- %f \n", *fp++ ) ;

		for ( j = 1 ; j < do_print ; j++ )
		{
			printf("%.2f	", *fp++ ) ;
			if (!(( j + 1 ) % 8 ))
				printf("\n") ;
		}
		printf("\n") ;

		if ( do_skip )
		{
			printf("after skip --- %d print %d\n	", do_skip, do_print2 ) ;

			fp = ( float *)oip + 3 + do_skip ; // is NUM_OF_HVT_INDEX

			for ( j = 0 ; j < do_print2 ; j++ )
			{
				printf("%.2f	", *fp++ ) ;
				if (!(( j + 1 ) % 8 ))
					printf("\n	") ;
			}
			printf("\n") ;
		}
	
		oip += record_size + 3 ;	// is NUM_OF_HVT_INDEX
	}
}

// start is to skip a number of rows
void
cs_p_d_tvh ( const char *s, int *dmem, int record_size, int num_rec, int do_print, int do_skip,
	int start )
{
	printf("%s ::: dmem %p record_size %d num_rec %d doprint %d start %d skip %d\n",
		s, dmem, record_size, num_rec, do_print, start, do_skip ) ;

	dmem += ( start * ( record_size + 3 )) ;

	cs_p_d_tvh ( s, dmem, record_size, num_rec, do_print, do_skip ) ;
}
