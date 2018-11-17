#ifndef __CS_RECON_H__
#define __CS_RECON_H__

// hdeder from client to server 

struct recon_param {
	int wht_size ;	// wht_side
	int r ;	// original
	int c ; 
	int iter ; 
	float lambda ;  
	float TVweight ; 
	int r_start ;	// offset to real image
	int	c_start ;
	int sel_idx ;
	int sel_size ; // selection
	int size ; // inbyte ... measurements
} ;

// header from server to client 

struct bits_to_client {
	int tag ;
	int col ;
	int row ;
	int size ;
	int format ;
	int t1 ;
	int t2 ;
	int t3 ;
} ;

// bits_to_client.tags
#define TAG_1	0xdeadbeef

// bits_to_client.format
#define FORMAT_RGB		1
#define FORMAT_JPG		2

// proto

unsigned char * reconstruct( int *meap, struct recon_param *pp, int *outsize ) ;

#define RECON_DBG

#ifdef RECON_DBG 

float * A( float *ip, int wht_size, int *sel_idx, int idx_size ) ;
template<typename T> float * At( T *ip, int wht_size, int *sel_idx, int idx_size ) ;
float * TV_GAP_rgb_use( int *ip, struct recon_param *para ) ;

int dv( float *ffp, float *ttp, int col, int row, int do_t ) ;
int dh( float *ffp, float *ttp, int col, int row, int do_t ) ;
int dvt ( float *ffp, float *ttp, int col, int row ) ;
int dht ( float *ffp, float *ttp, int col, int row ) ;
int clip ( float *ffp, float *ttp, int size, float lambda ) ;
int TV_denoising( float *y0, float *x0, float lambda, int col, int row, int iter ) ;

void i_recon_set_dbg( int *p ) ;

#endif 

#endif 


