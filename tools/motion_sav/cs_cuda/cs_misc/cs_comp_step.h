#ifndef __CS_COMP_STEP_H__
#define __CS_COMP_STEP_H__

/* in solveExact.m 
	step = comp_step(drct, Pdrct, xerr, lambda);

if ~init_cg
	gLd = beta.D * dot(Pgrad_x.D, Pdrct.D) + ...
	beta.scldA*dot(Pgrad_x.A,Pdrct.A);
	past_wgt = gLd/dLd;
	dLd = gLg  + past_wgt*(-2*gLd + past_wgt*dLd);

	% Update the direction drct based on the new gradient
	drct = - grad_x + past_wgt*drct;
	Pdrct.A = -Pgrad_x.A + past_wgt*Pdrct.A;
	Pdrct.D = -Pgrad_x.D + past_wgt*Pdrct.D;
else
	dLd = gLg;
	drct = -grad_x;
	Pdrct = struct('A', -Pgrad_x.A, 'D', -Pgrad_x.D);
	init_cg = false;
end
*/

struct cs_step {
	float LA ;
	float LD ;
	float EA ;
    float ED ;
	float A2 ;
	float D2 ;

	float *d_X ;	// 38016 ... v*h*t*c
	float *d_A ;	// 4752 ... should have v*h*t*c in size
	float *d_D ;	// 75072 ... v*h*t*c x2
} ;

void h_comp_step ( struct cs_step *d_cs_step, struct xerr *d_xerrp,
	struct Pgrad_x *d_lambdap, float *d_tmp, struct vhtc *h_vhtcp ) ;

float h_move_step ( float *d_xvec, struct xerr *d_xerrp, struct cs_step *d_cs_step,
	float alpha, struct sqrerr *sqrp, struct lmderr *lmderrp, struct beta *betap,
	float *d_wvec_refp, float *d_tmp, float *d_tmp1, float *d_tmp2, struct vhtc *vhtcp,
	struct lambda *lambdap ) ;

void h_update_lambda( struct beta *betap, struct lambda *lambdap, struct lmderr *lmderrp,
	struct vhtc *vhtcp, struct grad_x_const *gxcp, struct xerr *xerrp,
	struct cjerr *cjerrp, float *grad_xp, float *d_tmp ) ;

#endif 
