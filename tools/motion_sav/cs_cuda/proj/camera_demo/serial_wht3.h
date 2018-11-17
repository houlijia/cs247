#ifndef __SERIAL_WHT3_H__
#define __SERIAL_WHT3_H__

// #define CUDA_DBG 

int max_log2( int i )  ;

void reshape ( int *tp, int *fp, int size ) ;

void p_num_nm ( const char *s, int *dp, int col, int row ) ;
void p_num_nm_f ( const char *s, float *dp, int col, int row ) ;
void p_num_nm_uc ( const char *s, unsigned char *dp, int col, int row ) ;
void p_num( const char *s, int *fp, int cnt ) ;
void p_num_f( const char *s, float *fp, int cnt ) ;

template<typename T> void mea_un_select ( float *tp, T *fp, int total_size, int *sel_idx, int idx_size ) ;
template<typename T> void mea_select ( T *tp, T *fp, int *sel_idx, int idx_size ) ;

template<typename T> T* wht( T *to_datap, T *datap, int wht_size ) ;

template<typename T> void p_num_nm_x ( const char *s, T *dp, int col, int row ) ;
#endif 
