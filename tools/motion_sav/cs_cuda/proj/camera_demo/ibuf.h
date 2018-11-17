#ifndef __I_BUF_H__
#define __I_BUF_H__

void buf_swap (float **fp1, float **fp2 ) ;

void buf_p( const char *s ) ;

char * buf_get() ;

void buf_put( char *p ) ;

int buf_init( int size, int num ) ;

#endif 
