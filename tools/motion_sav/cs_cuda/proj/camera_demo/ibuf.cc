#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "serial_wht3.h"

struct buf_desc {
 	char *buf ;
	struct buf_desc *np ;
} ;

static struct buf_desc *buf_head = NULL ;
static struct buf_desc *free_head = NULL ;
static int buf_cnt ;
static int max_free ;

void
buf_swap (float **fp1, float **fp2 )
{
	float *fp ;

	fp = *fp1 ;
	*fp1 = *fp2 ;
	*fp2 = fp ;
}

void
buf_p( const char *s )
{
	printf("%s : %s ==== : head %p free %p cnt %d max_free %d\n", __func__, s,
	       (void*)buf_head, (void*)free_head, buf_cnt, max_free ) ;

#ifdef CUDA_OBS 

	struct buf_desc *dp = free_head ;
	while ( dp != NULL )
	{
		printf("free head %p : np %p bp %p \n", dp, dp->np, dp->buf ) ;
		dp = dp->np ;
	}

	dp = buf_head ;
	while ( dp != NULL )
	{
		printf("ready %p : np %p bp %p \n", dp, dp->np, dp->buf ) ;
		dp = dp->np ;
	}
#endif 
}

char *
buf_get()
{
	char *bp = NULL ;
	struct buf_desc *dp ;

#ifdef CUDA_OBS 
	buf_p("buf_get ------------------") ;
#endif 

	if ( buf_cnt )
	{
		buf_cnt-- ;
		if ( max_free > buf_cnt )
			max_free = buf_cnt ;

		dp = buf_head ;
		buf_head = buf_head->np ;

		bp = dp->buf ;

		dp->np = free_head ;
		dp->buf = NULL ;

		free_head = dp ;
	} else
		printf("%s : err : no more cnt %d max_free %d\n", __func__, buf_cnt, max_free ) ;

#ifdef CUDA_OBS 
	printf("%s : GOT %p \n", __func__, bp ) ;
#endif 

	return ( bp ) ;
}

void
buf_put( char *p )
{
	struct buf_desc *dp ;

#ifdef CUDA_OBS 
	printf("%s : %p\n", __func__, p) ;
	buf_p("buf_put ******************" ) ;
#endif 

	if ( free_head )
	{
		buf_cnt++ ;
		dp = free_head ;
		free_head = free_head->np ;

		dp->buf = p ;
		dp->np = buf_head ;

		buf_head = dp ;
	} else
		printf("%s : err : no free buf cnt %d max_free %d\n", __func__, buf_cnt, max_free ) ;

}

void
buf_dbg( const char *s )
{
	struct buf_desc *bp ;

	buf_p( s ) ;

	bp = buf_head ;
	while ( bp != NULL )
	{
	  printf("bp %p buf %p\n", (void*)bp, (void*)bp->buf ) ;
		// p_num ("buf",( int *)( bp->buf ), 64 ) ;
		bp = bp->np ;
	}

	bp = free_head ;
	while ( bp != NULL )
	{
	  printf("free dp %p \n", (void*)bp ) ;
		bp = bp->np ;
	}
}

int
buf_init( int size, int num )
{
	int i ;
	char *cp ;
	struct buf_desc *bp ;

	printf("%s : size %d num %d \n", __func__, size, num ) ;

	buf_cnt = 0 ;
	max_free = num ;

	i = num ;
	free_head = NULL ;
	while ( i-- )
	{
		bp = ( struct buf_desc * )malloc ( sizeof ( *bp )) ; 

#ifdef CUDA_OBS 
		printf(" ... i %d bp %p \n", i, bp ) ;
#endif 

		if ( bp == NULL )
			return ( 0 ) ;

		bp->np = free_head ;
		bp->buf = NULL ;

		free_head = bp ;
	}

	i = num ;
	while ( i-- )
	{
		cp = ( char * )malloc ( size ) ; 

#ifdef CUDA_OBS 
		printf(" ... i %d cp %p \n", i, cp ) ;
#endif 

		if ( cp == NULL )
			return ( 0 ) ;

#ifdef CUDA_OBS 
		memset ( cp, i, size ) ;
#endif 
		buf_put ( cp ) ;
	}

	printf("%s : done size %d cnt %d head %p free %p cnt %d max %d \n", __func__,
	       size, num, (void*)buf_head, (void*)free_head, buf_cnt, max_free ) ;

#ifdef CUDA_OBS 
	buf_dbg("init") ;
#endif 

	return ( 1 ) ;
}
