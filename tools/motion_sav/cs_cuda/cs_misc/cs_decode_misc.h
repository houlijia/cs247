#ifndef __CS_DECODE_MISC_H__
#define __CS_DECODE_MISC_H__

struct grad_x_const {
	float *d_A ;	 // 38016 == 72 * 88 * 2 * 3
	float *d_D ;	 // 38016 == 72 * 88 * 2 * 3
	float *d_sum ; 	 // 38016 == 72 * 88 * 2 * 3
} ;

struct Pgrad_x {
	float *d_A ; // 4572 ... but need v*h*t*c*2
	int A_size ;
	float *d_D ; // 75072 in .m ... in .cu it is v*h*t*c*2
	int D_size ;
} ;

struct vhtc {
	int v ;
	int h ;
	int t ;
	int c ;

	int size ; // v * h * t * c
	int size_mod2 ;
	int size_x2 ; 	

	int A_size ; // 4752
	int D_size ; // 75072
} ;

struct final {
	float scldA ;
	float A ;
	float D ;
} ;

struct beta {
	float A ;
	float D ;
	// float final ;
	struct final final ;
	float scldA ;
} ;

struct LRperms {	
	RndC_uint32 *d_Lperm ; // 65536, max_log2( 72x88x2x3 )
	RndC_uint32 *d_Rperm ; // 65536, max_log2( 72x88x2x3 )
} ;

struct cjerr {	
	float *d_A ; // 38016 ... 72x88x2 x3
	float *d_D ; // 38016 ... 72x88x2 x3
} ;

#endif 
