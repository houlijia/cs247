#ifndef __CS_DBG_H__ 
#define __CS_DBG_H__ 

#include "cs_header.h"

// #define CUDA_DBG 

int dbg_init( int ) ;
void dbg_clear_buf( int *cp, int size ) ;
void dbg_set_buf( int *cp, int size, int set ) ;
int dbg_put_d_data ( char *dp, char *hp, int size )  ;
int dbg_get_d_data ( char *dp, char *hp, int size )  ;
void dbg_pdata_i( char *s, int *dp, int size ) ;
void dbg_pdata_c( char *s, char *dp, int size ) ;
void dbg_mdata( int *dp, int size ) ;
void dbg_p_d_data_c ( char *s, char *dp, int size ) ;
void dbg_p_d_data_i ( char *s, int *dp, int size ) ;
char *dbg_d_malloc_c ( int size ) ;
int *dbg_d_malloc_i ( int size ) ;
void dbg_p_d_data_i_mn_skip ( char *s, int *dp, int size, int m, int n, int z,
	int doprint, int perm_size ) ;

void dbg_p_d_data_i_mn ( char *s, int *dp, int size, int m, int n, int doprint ) ;
void dbg_p_d_data_c_mn ( char *s, char *dp, int size, int m, int n, int doprint ) ;
void dbg_p_data_i_mn ( char *s, int *dp, int size, int m, int n, int doprint ) ;

void dbg_p_d_data_i_mn_v2 ( char *s, int *devp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y ) ;
void dbg_p_data_i_mn_v2 ( char *s, int *hp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y ) ;

// tsting ...
void dbg_p_d_data_ll ( char *s, long long *dp, int size ) ;

#endif 
