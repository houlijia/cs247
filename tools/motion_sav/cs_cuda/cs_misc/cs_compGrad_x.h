#ifndef __CS_COMPGRAD_X_H__
#define __CS_COMPGRAD_X_H__

void h_do_multVec( float *d_in, int in_size, float *d_out, int out_size,
	float *d_tmp, int size, 
	RndC_uint32 *d_Lperm, RndC_uint32 *d_Rperm,
	int *d_zeroed_rows, int num_zeroed_rows ) ;

void h_do_multTrnspVec( float *d_inout, float *d_tmp, RndC_uint32 *d_Lperm, RndC_uint32 *d_Rperm,
	int orig_size, int size, int final_size ) ;

void h_do_Grad_x( struct beta *betap, struct cjerr *cjerrp, 
	struct grad_x_const *gxcp, 
	float *d_grad_x, float *d_tmp1, float *d_tmp2, int size ) ;

void h_do_Pgrad_x( float *d_grad_x, int in_size, struct Pgrad_x *d_pgrad_xp,
	struct LRperms *lrpp, 
	int *d_zeroed_rows, int num_zeroed_rows,
	float *d_tmp, int size,
	struct vhtc *vhtcp ) ;

float h_do_gLg ( struct beta *betap, struct Pgrad_x *pgrad_x_p , float *d_tmp, struct vhtc *vhtcp ) ;

float h_updateGrad_x ( struct beta *betap, struct cjerr *cjerrp, struct grad_x_const *gxcp, 
	float *d_grad_xp, float *d_tmp1, float *d_tmp2, int size,
	struct Pgrad_x *pgrad_xp,
	struct LRperms *lrpp, 
	int *d_zeroed_rows, int num_zeroed_rows,
	struct vhtc *vhtcp ) ;

#endif 
