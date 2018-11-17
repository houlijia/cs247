#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_matrix.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_decode_misc.h"
#include "cs_complgrngn.h"

// #define CUDA_DBG
#define CUDA_DBG1

float
h_do_compLgrng ( struct beta *betap, struct lmderr *lmderrp, struct sqrerr *sqrerrp,
	struct xerr *xerrp )
{
	float lgr ;

	lgr = xerrp->J + lmderrp->D + betap->D * 0.5 * sqrerrp->D +
		lmderrp->A + betap->scldA * 0.5 * sqrerrp->A ;

	return ( lgr ) ;
}

int
h_do_sqrerr ( struct sqrerr *sqrerrp, struct xerr *xerrp, float *d_p, int tbl_size )
{
	if (( tbl_size < xerrp->A_size ) || ( tbl_size < xerrp->D_size ))
	{
		printf("%s: error tbl %d A %d D %d \n", __func__, tbl_size, xerrp->A_size, xerrp->D_size ) ;
		return ( 0 ) ;
	}

	sqrerrp->A = h_do_dot ( xerrp->d_A, xerrp->d_A, d_p, xerrp->A_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ( "h_do_sqrerr.A", d_p, xerrp->A_size );
#endif 

	sqrerrp->D = h_do_dot ( xerrp->d_D, xerrp->d_D, d_p, xerrp->D_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ( "h_do_sqrerr.D", d_p, xerrp->D_size );
#endif 

	return ( 1 ) ;
}

int
h_do_lmderr ( struct lmderr *lmderrp, struct lambda *lambdap, struct xerr *xerrp,
	float *d_p, int tbl_size )
{
	if (( tbl_size < xerrp->A_size ) || ( tbl_size < xerrp->D_size ) ||
		( lambdap->A_size != xerrp->A_size ) || ( lambdap->D_size != xerrp->D_size ))
	{
		printf("%s: error tbl %d xerr A %d D %d lambda A %d D %d\n", __func__, 
			tbl_size, xerrp->A_size, xerrp->D_size, lambdap->A_size, lambdap->D_size ) ;
		return ( 0 ) ;
	}

	lmderrp->A = h_do_dot ( lambdap->d_A, xerrp->d_A, d_p, xerrp->A_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ( "h_do_lmderr.A", d_p, xerrp->A_size );
#endif 

	lmderrp->D = h_do_dot ( lambdap->d_D, xerrp->d_D, d_p, xerrp->D_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f ( "h_do_lmderr.D", d_p, xerrp->D_size );
#endif 

	return ( 1 ) ;
}
