#ifndef __CS_HELPER_H__
#define __CS_HELPER_H__

int max_log2( int ) ;

// clear the device mem ... n in size of int
void clear_device_mem_i ( int *d_a, int n ) ;
void set_device_mem_i ( int *d_a, int n, int val ) ;

// clear the device mem ... n in size of char
void clear_device_mem_c ( char *d_a, int n ) ;
void set_device_mem_c ( char *d_a, int n, char c ) ;
void h_expand_c_to_i ( char *din, int *dout, int n ) ;	

// block adjustment
void h_block_adj( int n, int nThreadsPerBlock, int *block );
//

// omp timer 
int omp_timer_init( int ) ;
int omp_timer_on( int ) ;
int omp_timer_off( int ) ;
int omp_timer_get( int, double *dp, int *cp, double *ddp ) ;

// opt
int get_nums( int ac, char *av[], int idx, int cnt, int *np ) ;
unsigned int htoi(const char s[]) ;

// cs timer 
int cs_timer_init( int cnt ) ;
int cs_timer_on( int idx ) ;
int cs_timer_off( int idx ) ;
int cs_timer_get( int idx, clock_t *sp, clock_t *up, int *cp,
	double *asp, double *aup ) ;

// swap
void htonl_device_mem_i ( int *d_a, int n )	;

// tsting ...
void h_tst_longlong ( long long *d_a, int n )	 ;

// abs a vector

int h_set_abs ( int *d_a, int n, int record_len, int skip )	 ;

// set config
void h_set_config( struct cs_xyz *d_a, struct cube *hp ) ;

// weight shift
int weight_sft( int scheme, int size, int m, int n ) ;

// generic read/write int from/to device ... for ret code
int put_d_data_i ( int *dp, int *hp, int size ) ;
int get_d_data_i ( int *dp, int *hp, int size ) ;

#endif 
