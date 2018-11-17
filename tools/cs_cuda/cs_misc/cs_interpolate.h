#ifndef __CS_INTERPOLATE_H__
#define __CS_INTERPOLATE_H__

#define INT_YUV420 	0x3			// data in yuv style

int
h_make_interpolate ( int *d_input, int *d_output,
	int xdim, int ydim, int zdim, int scheme
#ifdef CUDA_OBS 
	, int *cudadbgp 
#endif 
	) ;

#endif 
