#ifndef __CS_MATRIX_H__
#define __CS_MATRIX_H__

int h_do_vector_add_destroy ( int *d_in, int tbl_size ) ;
float h_do_vector_add_destroy ( float *d_in, int tbl_size ) ;

int h_do_dot ( int *d_input1, int *d_input2, int *d_tmp, int tbl_size ) ;
float h_do_dot ( float *d_input1, float *d_input2, float *d_tmp, int tbl_size ) ;

void h_do_scale_mul_vector( int *d_in1, int scale, int tbl_size ) ;
void h_do_scale_mul_vector( float *d_in1, float scale, int tbl_size, float *d_out ) ;

// void h_do_scale_add_vector( float *d_in1, float toadd, int tbl_size ) ;
template<typename T> void h_do_scale_add_vector( T *d_in1, T toadd, int tbl_size ) ;

void h_do_copy_vector( float *d_in1, float *d_out, int tbl_size ) ;

void h_do_abs_vector( float *d_in1, int tbl_size ) ;

void h_do_vector_sub_vector ( float *d_in1, float *d_in2, float *d_out, int tbl_size ) ;
void h_do_vector_add_vector ( float *d_in1, float *d_in2, float *d_out, int tbl_size ) ;

float h_do_vector_2_norm ( float *d_a, float *d_tmp, int size ) ;
float h_do_max_destroy ( float *d_in, int tbl_size ) ;
float h_do_vector_inf_norm ( float *d_a, float *d_tmp, int size ) ;

void h_do_vhtc_2_hvtc ( float *d_in, float *d_out, int v, int h, int t, int c ) ;
void h_do_hvtc_2_vhtc ( float *d_in, float *d_out, int v, int h, int t, int c ) ;

#endif 
