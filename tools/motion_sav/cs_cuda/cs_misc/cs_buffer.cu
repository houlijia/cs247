#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include <semaphore.h>

#include "cs_cuda.h"
#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_buffer.h"

// #define CUDA_DBG
// #define CUDA_DBG1

static int cs_buf_cnt ;
static int cs_buf_free_cnt = 0 ;

static struct cs_buf *cs_buf_free ;
static struct cs_buf_list *cs_buf_lp ;

static sem_t mutex ;
static int do_lock = 0 ;

void cs_put_free_buf ( struct cs_buf *bp ) ;
struct cs_buf * cs_get_free_buf () ;


// lock : 1->do semaphore for the critical section 
int
cs_buffer_init ( struct cs_buf_desc *cbp, int cnt, int lock )
{
	char *d_p ;
	int icnt, total, i, j ;
	struct cs_buf_desc *cp ;
	struct cs_buf *bp ;

	do_lock = lock ;

	if ( do_lock )
	{
		if ( sem_init ( &mutex, 0, 1 ) < 0 )
		{
			printf("%s : sem_init failed \n", __func__ ) ;
			return ( 0 ) ;
		}
	}

	cs_buf_cnt = cnt ;
	cp = cbp ;

	total = 0 ;
	icnt = 0 ;
	for ( i = 0 ; i < cnt ; i++ )
	{
		total += (((( cp->size - 1 ) >> 2 ) + 1 ) << 2 ) * cp->cnt ;  

		icnt += cp->cnt ;

#ifdef CUDA_DBG 
		printf("%s: i %d total %d size %d cnt %d total cnt %d\n", __func__, i,
			total, cp->size, cp->cnt, icnt ) ;
#endif 
		cp++ ;
	}

	if (( i = cudaMalloc ( &d_p, total )) != cudaSuccess )
	{
		printf("%s: malloc failed %d \n", __func__, i ) ;
		return ( 0 ) ;
	}

	if (!( cs_buf_lp = ( struct cs_buf_list *)malloc ( sizeof ( 
		struct cs_buf_list ) * cs_buf_cnt )))
	{
#ifdef CUDA_DBG 
		printf("%s: cs_list failed \n", __func__ ) ;
#endif 
		return ( 0 ) ;
	}

	if (!( bp = cs_buf_free = ( struct cs_buf *)malloc ( sizeof ( struct cs_buf ) * icnt )))
	{
#ifdef CUDA_DBG 
		printf("%s: cs_bufp failed \n", __func__ ) ;
#endif 
		return ( 0 ) ;
	}

	cs_buf_free = NULL ;

	while ( icnt-- )
	{
		cs_put_free_buf ( bp ) ;
		bp++ ;
	}

	cp = cbp ;
	for ( i = 0 ; i < cs_buf_cnt ; i++ )
	{
		cs_buf_lp[i].bp = NULL ;
		cs_buf_lp[i].cnt = 0 ;

		total = (((( cp->size - 1 ) >> 2 ) + 1 ) << 2 ) ;  
		for ( j = 0 ; j < cp->cnt ; j++ )
		{
			cs_put_free_list( d_p, i ) ;
			d_p += total ;
		}

		cs_buf_lp[i].used = 0 ;
		cs_buf_lp[i].max_used = 0 ;
			
		cp++ ;
	}

	p_buffer_dbg("init") ;

	return ( 1 ) ;
}

void
p_buf_free(const char *s )
{
	struct cs_buf *cb ;
	int i = 0 ;

	printf("%s: %s cnt %d \n", __func__, s, cs_buf_free_cnt ) ;

	if ( do_lock )
		sem_wait ( &mutex ) ;

	cb = cs_buf_free ;
	while ( cb )
	{
		i++ ;
		printf("%i :: %p \n", i, cb ) ;
		cb = cb->np ;
	}

	if ( do_lock )
		sem_post( &mutex ) ;
}

void
p_buf_list( const char *s )
{
	struct cs_buf *cb ;
	int i, j, k ;

#ifdef CUDA_DBG 
	printf("%s: %s cs_buf_cnt %d\n", __func__, s, cs_buf_cnt ) ;
#endif

	if ( do_lock )
		sem_wait ( &mutex ) ;

	for ( i = 0 ; i < cs_buf_cnt ; i++ )
	{
		cb = cs_buf_lp[i].bp ;
		j = cs_buf_lp[i].cnt ;

		printf("%s : idx %d used %d cnt %d max_used %d \n",
			__func__, i, cs_buf_lp[i].used, cs_buf_lp[i].cnt, cs_buf_lp[i].max_used ) ;
		
		for ( k = 0 ; k < j ; k++ )
		{
#ifdef CUDA_DBG 
			printf("i %d k %d cb %p d_A %p np %p \n", i, k, cb, cb->d_A, cb->np ) ;
#endif 
			cb = cb->np ;
		}
	}	

	if ( do_lock )
		sem_post( &mutex ) ;
}

void
p_buffer_dbg( const char *s )
{
#ifdef CUDA_DBG 
	printf("--------------------------------------------------------------- B\n") ;
	p_buf_free( s ) ;
	p_buf_list( s ) ;
	printf("--------------------------------------------------------------- E\n") ;
#endif 
}

char *
cs_get_free_list ( int idx )
{
	struct cs_buf *cs_bufp ;
	char *bp = NULL ;

#ifdef CUDA_DBG 
	printf("%s: idx %d \n", __func__, idx ) ;
#endif 

	if ( idx < 0 || idx >= cs_buf_cnt )
	{
#ifdef CUDA_DBG 
		printf("%s: wrong idx %d \n", __func__, idx ) ;
#endif 
		return ( NULL ) ;
	}

	if ( do_lock )
		sem_wait ( &mutex ) ;

	cs_bufp = cs_buf_lp[ idx ].bp ;

	if ( cs_bufp )
	{
		cs_buf_lp[ idx ].bp = cs_bufp->np ; 

		bp = cs_bufp->d_A ;

		cs_buf_lp[ idx ].cnt-- ;
		cs_buf_lp[ idx ].used++ ;

		if ( cs_buf_lp[ idx ].used > cs_buf_lp[ idx ].max_used )
			cs_buf_lp[ idx ].max_used++ ;
	} 

#ifdef CUDA_DBG 
	printf("%s: idx %d bp %p cnt %d\n", __func__, idx, bp, cs_buf_lp[idx].cnt ) ;
#endif 

	if ( do_lock )
		sem_post( &mutex ) ;

	if ( cs_bufp )
		cs_put_free_buf( cs_bufp ) ;

	if ( bp == NULL )
	{
		printf("buffer ERR : idx %d OUT OF BUFFER wait forever\n", idx) ;
		while ( 1 )
			sleep ( 60 ) ;
	}

#ifdef CUDA_OBS 
	p_buffer_dbg("cs_get_free_list") ;
#endif 
	return ( bp ) ;
}

void
cs_put_free_list ( char *bp, int idx )
{
	struct cs_buf *fbp ;

#ifdef CUDA_DBG 
	printf("%s: bp %p idx %d \n", __func__, bp, idx ) ;
#endif 

	if ( idx < 0 || idx >= cs_buf_cnt )
	{
#ifdef CUDA_DBG 
		printf("%s: wrong idx %d \n", __func__, idx ) ;
#endif 
		return ;
	}

	fbp = cs_get_free_buf() ;

	if ( fbp == NULL )
	{
		printf("%s: no free buf left idx %d bp %p\n", __func__, idx, bp ) ;
		return ;
	}

	if ( do_lock )
		sem_wait ( &mutex ) ;

	fbp->d_A = bp ;

	if ( cs_buf_lp[ idx ].bp )
		fbp->np = cs_buf_lp[ idx ].bp ;
	else
		fbp->np = NULL ;

	cs_buf_lp[ idx ].bp = fbp ;
	cs_buf_lp[ idx ].cnt++ ;
	cs_buf_lp[ idx ].used-- ;

#ifdef CUDA_DBG 
	printf("%s: bp %p d_p %p idx %d cnt %d\n", __func__, cs_buf_lp[idx].bp,
		cs_buf_lp[idx].bp->d_A, idx, cs_buf_lp[idx].cnt ) ;
#endif 

	if ( do_lock )
		sem_post( &mutex ) ;
}

struct cs_buf *
cs_get_free_buf ()
{
	struct cs_buf *bp = NULL ;

	if ( do_lock )
		sem_wait ( &mutex ) ;

	if ( cs_buf_free )
	{
		bp = cs_buf_free ;
		cs_buf_free = cs_buf_free->np ;
		cs_buf_free_cnt-- ;
	}

#ifdef CUDA_DBG 
	printf("%s: bp %d cnt %d \n", __func__, bp, cs_buf_free_cnt ) ;
#endif 

	if ( do_lock )
		sem_post( &mutex ) ;

	return ( bp ) ;
}

void
cs_put_free_buf ( struct cs_buf *bp )
{
	if ( do_lock )
		sem_wait ( &mutex ) ;

	bp->np = cs_buf_free ;
	cs_buf_free = bp ;
	cs_buf_free_cnt++ ;

#ifdef CUDA_DBG 
	printf("%s: bp %d cnt %d \n", __func__, bp, cs_buf_free_cnt ) ;
#endif 

	if ( do_lock )
		sem_post( &mutex ) ;
}

void
cs_buf_swap (float **fp1, float **fp2 )
{
	float *fp ;

	fp = *fp1 ;
	*fp1 = *fp2 ;
	*fp2 = fp ;
}

