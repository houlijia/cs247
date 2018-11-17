#ifndef __CS_COMPLGRNGN_H__
#define __CS_COMPLGRNGN_H__

struct sqrerr {
	float A ;		//||(Ax-b)||^2
	float D ;		//||(Dx-w)||^2
} ;

struct xerr {
	float *d_A ;	// 4752x1  (Ax-b)
	int A_size ;

	float *d_D ;	// 75072x1 (Dx-w)
	int D_size ;

	float J ;
} ;

struct lmderr {
	float A ;	// lambda_A'(Ax-b) == lambda_A' * xerr.d_A
	float D ;   // lambda_D'(Dx-w) == lambda_D' * xerr.d_D
} ;

struct lambda {
	float *d_A ;	// 4752x1
	int A_size ;

	float *d_D ;	// 75072x1
	int D_size ;
} ;

// protos

float h_do_compLgrng ( struct beta *betap, struct lmderr *lmderrp, struct sqrerr *sqrerrp,
	struct xerr *xerrp ) ;

int h_do_sqrerr ( struct sqrerr *sqrerrp, struct xerr *xerrp, float *d_p, int tbl_size ) ;
int h_do_lmderr ( struct lmderr *lmderrp, struct lambda *lambdap, struct xerr *xerrp,
	float *d_p, int tbl_size ) ;

#endif 
