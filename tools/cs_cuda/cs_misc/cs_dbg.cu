#include <stdio.h>
#include <errno.h>
#include "cs_header.h"

#include "cs_dbg.h"

#define CUDA_DBG 

#ifdef CUDA_DBG 

#define DBG_BUF_SIZE (1024 * 1024)
int *dbg_bufp, dbg_size ;

void dbg_pdata_ll( char *s, long long *dp, int size ) ;

int
dbg_init( int size )
{
	if (!( dbg_bufp = ( int * ) malloc ( size )))
	{
		fprintf( stderr, "dbg_init: malloc failed \n", errno ) ;
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
dbg_p_d_data_ll ( char *s, long long *dp, int size )
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
dbg_p_data_i_mn ( char *s, int *dp, int size, int m, int n, int doprint )
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
dbg_p_d_data_c_mn ( char *s, char *dp, int size, int m, int n, int doprint )
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
dbg_p_d_data_i_mn_skip ( char *s, int *dp, int size, int m, int n, int z, int doprint, int perm_size )
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
dbg_p_d_data_i_mn ( char *s, int *dp, int size, int m, int n, int doprint )
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
dbg_p_data_i_mn_v2 ( char *s, int *hp, int size, int doprint,
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

void
dbg_p_d_data_i_mn_v2 ( char *s, int *devp, int size, int doprint,
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

void
dbg_p_d_data_i ( char *s, int *dp, int size )
{
	size <<= 2 ;

	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "dbg_p_d_data: %s size %d dp %p\n", 
		s, size, dp ) ;

	dbg_get_d_data (( char *)dp, ( char *)dbg_bufp, size ) ;

	size >>= 2 ;

	dbg_pdata_i ( s, ( int *)dbg_bufp, size ) ; 
}

void
dbg_p_d_data_c ( char *s, char *dp, int size )
{
	if ( size > dbg_size )
		size = dbg_size ;
	
	fprintf( stderr, "dbg_p_d_data: %s size %d dp %p\n", 
		s, size, dp ) ;

	dbg_get_d_data ( dp, ( char *)dbg_bufp, size ) ;

	dbg_pdata_c ( s, ( char *)dbg_bufp, size ) ; 
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
		fprintf(stderr, "dbg_get_d_data: failed %d\n", i ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

void
dbg_pdata_ll( char *s, long long *dp, int size )
{
	int i ;

	fprintf( stderr, "dbg_pdata_ll: %s\n", s ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %p -- %d 0x%x\n", i, dp, *dp, *dp ) ;
		i++ ;
		dp++ ;
	}
}

void
dbg_pdata_i( char *s, int *dp, int size )
{
	int i ;

	fprintf( stderr, "dbg_pdata_i: %s\n", s ) ;
	for ( i = 0 ; i < size ; )
	{
		fprintf( stderr, "%d -- %8.8x %d\n", i, *dp, *dp ) ;
		i++ ;
		dp++ ;
	}
}

void
dbg_pdata_c( char *s, char *dp, int size )
{
	int i ;
	unsigned char *cp = ( unsigned char *)dp ;

	fprintf( stderr, "dbg_pdata_c: %s\n", s ) ;
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

#endif 
