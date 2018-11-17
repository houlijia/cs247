#ifndef __CS_DCT_H__
#define __CS_DCT_H__

int h_do_dct ( int *d_input, int *d_output, int xdim, int ydim, int zdim ) ;
int h_do_dct ( float *d_input, float *d_output, int xdim, int ydim, int zdim, int inverse ) ;

int h_do_dct_init () ;

#endif 
