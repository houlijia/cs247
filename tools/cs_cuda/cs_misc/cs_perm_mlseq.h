#ifndef __CS_PERM_MLSEQ_H__
#define __CS_PERM_MLSEQ_H__

/* size as in 2^size */
int permutation_load( int size, char *dirp, int *d_leftp,
	int *d_rightp ) ;

void h_do_permutation_R ( int *d_input, int *d_output,
	int *d_perm_tbl, int size, int tbl_size ) ;

void h_do_permutation_L ( int *d_input, int *d_output,
	int *d_perm_tbl, int size, int tbl_size ) ;

void h_do_permutation_Lv2 ( int *d_input, int *d_output,
	int *d_perm_tbl_i, int *d_perm_tbl_s, int *d_perm_tbl_c, int n, int tbl_size,
	int nblk_in_x, int nblk_in_y ) ;

int permutation_load_2( int size, char *fipL, char *fipR, int *d_leftp,
	int *d_rightp ) ;

void h_do_permutation_double ( int *d_first, int *d_second, int *d_final,
	int tbl_size ) ;

/*
void h_do_permutation_RR ( int *d_perm_tblR, int *d_perm_tblRR,
	int *d_perm_tblNRR, int tbl_size ) ;

void h_do_permutation_LL ( int *d_perm_tblL, int *d_perm_tblLL,
	int *d_perm_tblNLL, int tbl_size ) ;
*/

#endif 
