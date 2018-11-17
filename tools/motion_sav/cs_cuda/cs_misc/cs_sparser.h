#ifndef __CS_SPARSER_H__
#define __CS_SPARSER_H__

// H: x V:y T:z  


int h_compDiffTrnspPx1_v( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp ) ;
int h_compDiffTrnspPx1_h( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp ) ;

int h_compDiffInside_v( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp ) ;
int h_compDiffInside_h( float *d_in, float *d_out, int tsize, struct vhtc *vhtcp )  ;

int h_compSprsVec( float *d_input, float *d_tmp, float *dvh_output,
	struct vhtc *vhtcp, int tsize ) ;

int h_compSprsVecTrnsp( float *d_in, float *d_out, float *d_tmp, int size, struct vhtc *vhtcp ) ;

void h_do_max_sign_in_optimize( float *dvh_input, float beta_inv, int dvh_size ) ;

void h_optimize( float *dvh_input, float *dvh_out, float *lambda, float beta, int dvh_size ) ;

float h_do_Jopt( float *dvh_input, float *dvh_tmp, struct vhtc *vhtcp ) ;

void h_do_cleanup_total_value_vh_elements( float *dvh_input, struct vhtc *vhtcp, int do_v ) ;
void h_do_cleanup_total_value_vh_elements( float *dvh_input, struct vhtc *vhtcp ) ;

int h_optimize_solver_w( float *d_xvec, float beta_D, float *d_lambda_D, float *d_wvec_ref,
	float *J_opt, float *d_wvec, float *d_dxerr_D, float *d_tmp, float *d_Dxvec,
	struct vhtc *vhtcp ) ;

#endif 
