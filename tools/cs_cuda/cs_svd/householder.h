#ifndef __HOUSEHOLDER_H__
#define __HOUSEHOLDER_H__

int hh_get_norm( double *cp, int column_data_cnt, int row_cnt,
	double *normp ) ;
int hh_d11( double *cp, int column_data_cnt, int row_cnt, double *d11p ) ;
int hh_matrix_reduction( double *cp, int column_data_cnt, int column_cnt,
	int row_cnt ) ;
void hh_apply_fi( double *ovp, double *ocp, int offset, int cnt ) ;
void hh_apply_fi_R( double *ovp, double *ocp, int column_data_cnt, int column_cnt,
	int row_cnt ) ;
void hh_apply_fi_C( double *ovp, double *ocp, int column_data_cnt, int column_cnt,
	int row_cnt ) ;
int hh_make_v( double *vp, double *cp, int column_data_cnt,
	int row_cnt, double w11, double f1 ) ;

#define CUDA_DBG

#ifdef CUDA_DBG 
void p_double_mn( char *s, double *dp, int m, int n ) ;
void p_double( char *s, double *dp, int cnt ) ;
#endif 

#endif 
