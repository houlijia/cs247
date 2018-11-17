#ifndef __CS_PERM_GENERIC_H__
#define __CS_PERM_GENERIC_H__

void
h_do_permutation_generic_f1 ( int *d_input, int *d_output, int *d_perm_tbl,
	int tbl_size ) ;

void
h_do_permutation_generic_f2 ( int *d_input, int *d_output, int *d_perm_tbl,
	int tbl_size ) ;

void
h_do_permutation_generic_f1 ( float *d_input, float *d_output, int *d_perm_tbl,
	int tbl_size ) ;

void
h_do_permutation_generic_f2 ( float *d_input, float *d_output, int *d_perm_tbl,
	int tbl_size ) ;

void
h_do_permutation_generic_inverse ( int *d_output, int *d_perm_tbl,
	int tbl_size ) ;

#endif 
