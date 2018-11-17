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
#include "cs_vector.h"
#include "cs_perm_generic.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_sparser.h"
#include "cs_complgrngn.h"
#include "cs_buffer.h"
#include "cs_decode_misc.h"
#include "cs_comp_step.h"

#define CUDA_DBG
#define CUDA_DBG1

/*
NOTE: there is no corresponding test/cs_comp_step_test.cu for unit testing.
since the logic is straight forward.  so we do the test in integration time.
*/

/* 
h_comp_step:

NOTE:the creater of any d_D has to make sure the extra row/column are wiped.

	d_drct : 38016 ... v*h*c*t
	d_Pdrctp, d_xerrp, d_lambdap have the same size
	d_tmp : 75072 ... v*h*c*t x2 ... note ... the last column/row
		could be junk
*/

void
h_comp_step ( struct cs_step *d_cs_step, struct xerr *d_xerrp,
	struct Pgrad_x *d_lambdap, float *d_tmp, struct vhtc *h_vhtcp )
{
#ifdef CUDA_OBS 
	// should be the creator's job  i.e. h_compSprsVec()
	h_do_cleanup_total_vlaue_vh_elements( d_cs_step->d_D, h_vhtcp ) ;
	h_do_cleanup_total_vlaue_vh_elements( d_lambdap->d_D, h_vhtcp ) ;
#endif 

	d_cs_step->LA = h_do_dot ( d_lambdap->d_A, d_cs_step->d_A, d_tmp, h_vhtcp->A_size ) ;  
	d_cs_step->LD = h_do_dot ( d_lambdap->d_D, d_cs_step->d_D, d_tmp, h_vhtcp->size_x2 ) ;  

	d_cs_step->EA = h_do_dot ( d_xerrp->d_A, d_cs_step->d_A, d_tmp, h_vhtcp->A_size ) ;  
	d_cs_step->ED = h_do_dot ( d_xerrp->d_D, d_cs_step->d_D, d_tmp, h_vhtcp->size_x2 ) ;  

	d_cs_step->A2 = h_do_dot ( d_cs_step->d_A, d_cs_step->d_A, d_tmp, h_vhtcp->A_size ) ;  
	d_cs_step->D2 = h_do_dot ( d_cs_step->d_D, d_cs_step->d_D, d_tmp, h_vhtcp->size_x2 ) ;  

}

/*
h_move_step:
	d_xvec : v*h*t*c
	d_tmp1, d_tmp2 : 75072 .. v*h*t*c x2 ... to used as wvec and dxvec in h_optimize_solver_w

	alpha: step size
	
*/

float
h_move_step ( float *d_xvec, struct xerr *d_xerrp, struct cs_step *d_cs_step,
	float alpha, struct sqrerr *sqrp, struct lmderr *lmderrp, struct beta *betap,
	float *d_wvec_refp, float *d_tmp, float *d_tmp1, float *d_tmp2, struct vhtc *vhtcp,
	struct lambda *lambdap )
{
	float lgrng ;

	// xvec = xvec + alpha*step.X;
	h_do_scale_mul_vector ( d_xvec, alpha, vhtcp->size, d_tmp1 ) ; 
	h_do_vector_add_vector ( d_xvec, d_tmp1, d_xvec, vhtcp->size ) ; 

	// xerr.A = xerr.A + alpha*step.A;
	h_do_scale_mul_vector ( d_cs_step->d_A, alpha, vhtcp->A_size, d_tmp1 ) ; 
	h_do_vector_add_vector ( d_xerrp->d_A, d_tmp1, d_xerrp->d_A, vhtcp->A_size ) ; 

	// [xerr.D, xerr.J] = optimize_solver_w(xvec, sparser, beta.D, lambda.D, wvec_ref);
	h_optimize_solver_w( d_xvec, betap->D, lambdap->d_D, d_wvec_refp,
		&d_xerrp->J, d_tmp1, d_xerrp->d_D, d_tmp, d_tmp2, vhtcp ) ;

	// lmderr.A =  lmderr.A + alpha*step.LA;
	lmderrp->A = lmderrp->A + alpha * d_cs_step->LA ;

	// lmderr.D = dot(lambda.D,xerr.D);
	lmderrp->D = h_do_dot ( lambdap->d_D, d_xerrp->d_D, d_tmp, vhtcp->size_x2 ) ;

	// sqrerr.A = sqrerr.A + 2*alpha*step.EA + alpha*alpha*step.A2;
	sqrp->A = sqrp->A + 2 * alpha * d_cs_step->EA + alpha * alpha * d_cs_step->A2 ; 

	// sqrerr.D = dot(xerr.D,xerr.D);
	sqrp->D = h_do_dot ( d_xerrp->d_D, d_xerrp->d_D, d_tmp, vhtcp->size_x2 ) ;

	// lgrngn = compLgrngn(beta, [], xerr, lmderr, sqrerr);

	lgrng = h_do_compLgrng ( betap, lmderrp, sqrp, d_xerrp ) ;

	return ( lgrng ) ;
}

/* 
h_updatge_lambda:



	d_tmp : h*v*t*c 38016 x 2

*/

void
h_update_lambda( struct beta *betap, struct lambda *lambdap, struct lmderr *lmderrp,
	struct vhtc *vhtcp, struct grad_x_const *gxcp, struct xerr *xerrp,
	struct cjerr *cjerrp, float *grad_xp, float *d_tmp )
{

	// lambda.A = lambda.A + beta.scldA * xerr.A;
	h_do_scale_mul_vector ( xerrp->d_A, betap->scldA,  vhtcp->A_size, d_tmp ) ; 
	h_do_vector_add_vector ( lambdap->d_A, d_tmp, lambdap->d_A, vhtcp->A_size ) ; 

	// lambda.D = lambda.D + beta.D * xerr.D;
	h_do_scale_mul_vector ( xerrp->d_D, betap->D, vhtcp->D_size, d_tmp ) ; 
	h_do_vector_add_vector ( lambdap->d_D, d_tmp, lambdap->d_D, vhtcp->D_size ) ; 

	// lmderr = struct('A', dot(lambda.A, xerr.A), 'D', dot(lambda.D, xerr.D));
	lmderrp->A = h_do_dot ( lambdap->d_A, xerrp->d_A, d_tmp, vhtcp->A_size ) ;  
	lmderrp->D = h_do_dot ( lambdap->d_D, xerrp->d_D, d_tmp, vhtcp->D_size ) ;  

	// grad_x_const.A = grad_x_const.A + beta.A*cjerr.A;
	h_do_scale_mul_vector ( cjerrp->d_A, betap->A, vhtcp->A_size, d_tmp ) ; 
	h_do_vector_add_vector ( d_tmp, gxcp->d_A, gxcp->d_A, vhtcp->A_size ) ; 

	// grad_x_const.D = grad_x_const.D + beta.D*cjerr.D;
	h_do_scale_mul_vector ( cjerrp->d_D, betap->D, vhtcp->D_size, d_tmp ) ; 
	h_do_vector_add_vector ( d_tmp, gxcp->d_D, gxcp->d_D, vhtcp->D_size ) ; 

	// grad_x_const.sum = grad_x;

	h_do_copy_vector ( grad_xp, gxcp->d_sum, vhtcp->size ) ;
}
