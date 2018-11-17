#ifndef __CS_BUFFER_H__
#define __CS_BUFFER_H__

struct cs_buf_desc {
	int size ;	// number of bytes per buffer
	int unit_size ; // size in int and float ;
	int cnt ;	// number of such buffers
} ;

struct cs_buf {
	char *d_A ; // device address
	struct cs_buf *np ;
} ;

struct cs_buf_list {
	struct cs_buf *bp ;
	int cnt ;

	int max_used ;
	int used ;
} ;

int cs_buffer_init ( struct cs_buf_desc *cbp, int cnt, int lock ) ;

// for debug purpose

void p_buffer_dbg( const char * ) ;

char * cs_get_free_list ( int idx ) ;
void cs_put_free_list ( char *bp, int idx ) ;

void cs_buf_swap (float **fp1, float **fp2 ) ;

#endif 
