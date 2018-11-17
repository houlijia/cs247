#ifndef __CS_DOMULTIVEC_H__
#define __CS_DOMULTIVEC_H__

void
h_do_multi_trnsp_vec ( float *d_input, float *d_output, int *d_R_perm_tbl, int *d_L_perm_tbl,
	int tbl_size ) ;

#ifdef CUDA_OBS 
void
h_do_multi_vec ( float *d_input, float *d_output, int *d_R_perm_tbl, int *d_L_perm_tbl,
	int tbl_size ) ;
#endif 

void
h_do_multi_vec ( float *d_input, float *d_output, float *d_tmp, int *d_R_perm_tbl,
	int *d_L_perm_tbl, int tbl_size, int keep ) ;

#endif 
