#ifndef __CS_DBG_H__ 
#define __CS_DBG_H__ 

#include "cs_header.h"

// #define CUDA_DBG 

int dbg_init( int ) ;
void dbg_clear_buf( int *cp, int size ) ;
void dbg_set_buf( int *cp, int size, int set ) ;
int dbg_put_d_data ( char *dp, char *hp, int size )  ;
int dbg_get_d_data ( char *dp, char *hp, int size )  ;
int dbg_copy_d_data ( char *dtp, char *dfp, int size ) ;
void dbg_pdata_i( const char *s, const int *dp, int size ) ;
void dbg_pdata_f( const char *s, const float *dp, int size ) ;
void dbg_pdata_c( const char *s, const char *dp, int size ) ;
void dbg_pdata_d( const char *s, const float *dp, int size ) ;
void dbg_mdata( int *dp, int size ) ;
void dbg_p_d_data_c ( const char *s, char *dp, int size ) ;
void dbg_p_d_data_i ( const char *s, int *dp, int size ) ;
void dbg_p_d_data_f ( const char *s, float *dp, int size ) ;
void dbg_p_d_data_d ( const char *s, float *dp, int size ) ;
char *dbg_d_malloc_c ( int size ) ;
int *dbg_d_malloc_i ( int size ) ;
float *dbg_d_malloc_f ( int size ) ;

void dbg_p_d_data_f_mn_skip ( const char *s, float *dp, int size, int m, int n, int z,
	int doprint, int perm_size ) ;
void dbg_p_d_data_i_mn_skip ( const char *s, int *dp, int size, int m, int n, int z,
	int doprint, int perm_size ) ;

void dbg_p_d_data_f_mn ( const char *s, float *dp, int size, int m, int n, int doprint ) ;
void dbg_p_d_data_f_mn ( const char *s, float *dp, int size, int m, int n, int doprint, int printrow ) ;
void dbg_p_d_data_i_mn ( const char *s, int *dp, int size, int m, int n, int doprint ) ;
void dbg_p_d_data_c_mn ( const char *s, char *dp, int size, int m, int n, int doprint ) ;
void dbg_p_data_i_mn ( const char *s, int *dp, int size, int m, int n, int doprint ) ;

// print the motion detection T/V/H/Va format
void dbg_p_data_md_f_mn ( const char *s, int *dp, int size, int m, int n, int doprint ) ;

void dbg_p_d_data_f_mn_v2 ( const char *s, float *devp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y ) ;
void dbg_p_d_data_i_mn_v2 ( const char *s, int *devp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y ) ;
void dbg_p_data_i_mn_v2 ( const char *s, int *hp, int size, int doprint,
	struct cube *dp, int blk_in_x, int blk_in_y ) ;

void dbg_p_d_data_i_cube ( const char *s, int *devp, int vx, int hy, int tz ) ;
void dbg_p_d_data_i_cube ( const char *s, float *devp, int vx, int hy, int tz ) ;
void dbg_p_data_i_cube ( const char *s, float *dp, int vx, int hy, int tz ) ;

void dbg_pr_first_last ( char *s, float *d_p, int len, int pr_size ) ;
void dbg_pr_first_last ( char *s, int *d_p, int len, int pr_size ) ;
void dbg_pr_h_first_last ( char *s, int *d_p, int len, int pr_size ) ;
void dbg_pr_h_first_last ( char *s, char *h_p, int len, int pr_size ) ;

// tsting ...
void dbg_p_d_data_ll ( const char *s, long long *dp, int size ) ;

// for random
int dbg_ck_unique ( char *s, int *dp, int size ) ;
int dbg_perm_ck ( int callid, int callid2, int zero_in_zero, int *d_bp, int size ) ;

// TVH
void cs_p_d_tvh ( const char *s, int *dmem, int record_size, int num_rec, int do_print, int do_skip ) ;
void cs_p_d_tvh ( const char *s, int *dmem, int record_size, int num_rec, int do_print,
	int do_skip, int start ) ;

#endif 
