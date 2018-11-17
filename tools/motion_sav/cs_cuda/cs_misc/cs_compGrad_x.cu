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
#include "cs_whm_encode_b.h"
#include "cs_matrix.h"
#include "cs_vector.h"
#include "cs_perm_generic.h"

#include "RndCState.h"
#include "RndC_ifc.h"

#include "cs_decode_misc.h"
#include "cs_sparser.h"
#include "cs_compGrad_x.h"

#define CUDA_DBG
#define CUDA_DBG1

/* 
   LDL ... working on this ... doing the sense.multTrnspVec ... should we move it to
   whm directory??
*/

/*
h_do_multVec:  cf. sensingmatrixsqr.m:doMultVec

	d_in :  
	in_size : 38016, 72x88x2 x3 // valid data size ... real size is mod 2
	d_out :
	out_size : 4752

	size : max_log2(in_size, i.e.38016 ... 72x88x2 x3 ) ... i.e. 65536 ... 
		compression ratio is 8:1 in this case

		size applies to d_in, d_out, d_tmp, d_Lperm and d_Rperm ... due to permutation

	NOTE: the rest of the entries in input d_out, i.e. entries beyond 4752, HAVE TO be 0
*/

void
h_do_multVec( float *d_in, int in_size, float *d_out, int out_size,
	float *d_tmp, int size, 
	RndC_uint32 *d_Lperm, RndC_uint32 *d_Rperm,
	int *d_zeroed_rows, int num_zeroed_rows )
{
#ifdef CUDA_DBG 
	printf("%s: in %p in_size %d out %p outsize %d tmp %p size %d\n",
		__func__, d_in, in_size, d_out, out_size, d_tmp, size ) ;

	printf("	Lp %p Rp %p row %p num %d\n",
		d_Lperm, d_Rperm, d_zeroed_rows, num_zeroed_rows ) ;

#endif 

	set_device_mem ( d_in + in_size, size - in_size, 0.0 ) ;

	h_do_permutation_generic_f1( d_in, d_tmp, ( int *)d_Rperm, size ) ;

	cs_whm_measurement_b( d_tmp, size, size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_multVec: after whm", d_tmp, max( size, 128 )) ;
#endif 

	h_do_permutation_generic_f2( d_tmp, d_out, ( int *)d_Lperm, size ) ;	// d_tmp: 65536 d_inout: 4752

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_multVec: after f2", d_out, max( size, 128 )) ;
#endif 

	// the data is in d_inout ... but only the first final size ( 4752 ) will be used

	set_device_mem ( d_out + out_size, size - out_size, 0.0 ) ;

	// apply the "zeroed_rows" here ...

	if ( num_zeroed_rows )
		h_do_vector_zero_some( d_out, d_zeroed_rows, num_zeroed_rows ) ;
}

/*
h_do_multTrnspVec:  cf. sensingmatrixsqr.m:doMultTrnspVec

	d_inout : the fist 4752 entries ( Dx -b ) has data				
	in_size : 4752
	size : max_log2(38016 ... 72x88x2 x3) i.e. 65536 ... 
		compression ratio is 8:1 in this case

		size applies to d_inout, d_tmp, d_Lperm and d_Rperm
	final_size : 38016 ... 72x88x2 x3

	NOTE: the rest of the entries in input d_inout, i.e. entries beyond 4752, HAVE TO be 0
	NOTE: the contents in d_inout is gone.  the output is in d_inout
*/

void
h_do_multTrnspVec( float *d_inout, float *d_tmp, RndC_uint32 *d_Lperm, RndC_uint32 *d_Rperm,
	int orig_size, int size, int final_size )
{
#ifdef CUDA_DBG 
	printf("%s: in %p tmp %p L %p R %p orig_size %d size %d final %d \n", 
		__func__, d_inout, d_tmp, d_Lperm, d_Rperm,
		orig_size, size, final_size ) ;
#endif 

	set_device_mem ( d_inout + orig_size, size - orig_size, 0.0 ) ;	// done by caller?

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_multTrnspVec: after set",d_inout, max( size, 128 )) ;
#endif 

	h_do_permutation_generic_f1( d_inout, d_tmp, ( int * )d_Lperm, size ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f("h_do_multTrnspVec: after f1",d_tmp, max( size, 128 )) ;
#endif 

	cs_iwhm_measurement_b( d_tmp, size, size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_multTrnspVec: after whm",d_tmp, max( size, 128 )) ;
#endif 

	h_do_permutation_generic_f2( d_tmp, d_inout, ( int * )d_Rperm, size ) ;	// d_tmp: 65536 d_inout: 38016

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_multTrnspVec: after f2",d_inout, max( size, 128 )) ;
#endif 

	// the data is in d_inout ... but only the first 38016 ( 72x88x2 x3 ) will be used

	set_device_mem ( d_inout + final_size, size - final_size, 0.0 ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("h_do_multTrnspVec: after set-2",d_inout, max( size, 128 )) ;
#endif 

}


/*
	h_do_Grad_x:	sizes are the minimum sizes 
		beta_A, beta_D : scalars
		d_cjerr_A, d_cjerr_D : 72x88x2 x3 ...
		d_grad_x_const_sum : 72x88x2 x3 ...
		size : 72x88x2 x3
		d_grad_x : 72x88x2 x3 ...	// this is the output 
		d_tmp1, d_tmp2 : 72x88x2 x3 ... 
*/

void
h_do_Grad_x( struct beta *betap, struct cjerr *cjerrp, 
	struct grad_x_const *gxcp, 
	float *d_grad_x, float *d_tmp1, float *d_tmp2, int size )
{
	// cf soveExact.m:updateGrad_x() step one ...
	// calculate grad_x

	h_do_scale_mul_vector( cjerrp->d_A, betap->A, size, d_tmp1 ) ;
	h_do_scale_mul_vector( cjerrp->d_D, betap->D, size, d_tmp2 ) ;

	h_do_vector_add_vector( d_tmp1, d_tmp2, d_tmp1, size ) ;

	h_do_vector_add_vector( d_tmp1, gxcp->d_sum, d_grad_x, size ) ;
}

/*
	h_do_Pgrad_x:
		d_grad_x : 72x88x2 x3 ... 38016 in 65536
		in_size : 38016 of d_grad_x
		size : max_log2 of in_size (38016 ... ) // 65536
		d_pgrad_xp->A_size: output size of h_do_multVec	// 4752
			max 38016, 72x88x2 x3 ... i.e. no compression 
			but buffer still has max_log2, due to permutation
		d_pgrad_xp->d_A : max 38016 when no compression
		d_pgrad_xp->d_D : 38016 x 2
		d_pgrad_xp->D_size : 38016 x 2
		d_Lperm, d_Rperm : max_log2( 38016 ... ) 65536
		d_tmp : same as size max_log2 of in_size, 65536

		OUTPUT are d_pgrad_xp->d_A and d_pgrad_xp->d_D
*/	

void
h_do_Pgrad_x( float *d_grad_x, int in_size, struct Pgrad_x *d_pgrad_xp,
	struct LRperms *lrpp, 
	int *d_zeroed_rows, int num_zeroed_rows,
	float *d_tmp, int size,
	struct vhtc *vhtcp )
{
	// cf soveExact.m:updateGrad_x()

	// h_do_copy_vector ( d_grad_x, d_tmp1, size ) ;	// can we avoid this ???	d_grad_x : 38016

	// calculate Pgrad_x

	// h_do_multVec will do the memory cleaning ...

	h_do_multVec( d_grad_x, in_size, d_pgrad_xp->d_A, d_pgrad_xp->A_size,
		d_tmp, size, 
		lrpp->d_Lperm, lrpp->d_Rperm,
		d_zeroed_rows, num_zeroed_rows ) ;

	h_compSprsVec( d_grad_x, d_tmp, d_pgrad_xp->d_D, vhtcp, vhtcp->size ) ;
}

/*
	h_do_gLg:
		pgrad_x_A_size : 4752 
		pgrad_x_D_size : 72x88x2 x3 x2 ... 75072 ... 
		d_tmp : same size as pgrad_x_D_size

	OUTPUT : the gLg returned

*/	

float
h_do_gLg ( struct beta *betap, struct Pgrad_x *pgrad_x_p , float *d_tmp, struct vhtc *vhtcp )
{
	float f, ff, ftmp ;

#ifdef CUDA_OBS 
	// no need ... h_compSprsVec() will wipe them
	h_do_cleanup_total_value_vh_elements( pgrad_x_p->d_D, vhtcp, 1 ) ;
	h_do_cleanup_total_value_vh_elements( pgrad_x_p->d_D + vhtcp->size , vhtcp, 0 ) ;
#endif 
	
	f = h_do_dot ( pgrad_x_p->d_A, pgrad_x_p->d_A, d_tmp, pgrad_x_p->A_size ) ;
	ff = h_do_dot ( pgrad_x_p->d_D, pgrad_x_p->d_D, d_tmp, pgrad_x_p->D_size ) ;

#ifdef CUDA_DBG 
	printf("%s : f %f ff %f scld %f D %f\n", __func__, f, ff, betap->scldA, betap->D ) ; 
#endif 

	ftmp = betap->scldA * f +  betap->D * ff ;

	return ( ftmp ) ;
}

float
h_updateGrad_x ( struct beta *betap, struct cjerr *cjerrp, struct grad_x_const *gxcp, 
	float *d_grad_xp, float *d_tmp1, float *d_tmp2, int size,
	struct Pgrad_x *pgrad_xp,
	struct LRperms *lrpp, 
	int *d_zeroed_rows, int num_zeroed_rows,
	struct vhtc *vhtcp )
{
	float gLg ;

#ifdef CUDA_DBG 
	printf("%s : before h_do_Grad_x ===================================\n", __func__ ) ;
#endif 

	h_do_Grad_x( betap, cjerrp, gxcp, d_grad_xp, d_tmp1, d_tmp2, vhtcp->size ) ;

#ifdef CUDA_DBG 
	printf("%s : before h_do_Pgrad_x ===================================\n", __func__ ) ;
#endif 

	h_do_Pgrad_x( d_grad_xp, size, pgrad_xp,
		lrpp, 
		d_zeroed_rows, num_zeroed_rows,
		d_tmp1, size, 
		vhtcp ) ;

#ifdef CUDA_DBG 
	printf("%s : before gLg ===================================\n", __func__ ) ;
#endif 

	gLg = h_do_gLg ( betap, pgrad_xp, d_tmp1, vhtcp ) ;

	return ( gLg ) ;
}
