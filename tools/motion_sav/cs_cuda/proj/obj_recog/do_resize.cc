#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <serial_wht3.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "do_resize.h"
#include "../../proj/camera_demo/serial_wht3.h"

#define CUDA_DBG

// front 1 back 2
#define WIDTH_ADJ		2
// top 1 bottom 2
#define HEIGHT_ADJ		2

float 
cubicInterpolate (float p[4], float x) 
{
	float f ;

	f = p[1] + 0.5 * x*(p[2] - p[0] + 
		x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])));

	// printf("%s : %f %f %f %f -- D %f == f %f\n", __func__, p[0], p[1], p[2], p[3], x, f ) ; 

	return ( f ) ;
}

float 
bicubicInterpolate (float *p, float x, float y, int width ) 
{
	float arr[4];

	// printf("%s :: x %f y %f width %d \n", __func__, x, y, width ) ;

	arr[0] = cubicInterpolate(p, y);
	arr[1] = cubicInterpolate(p+width, y);
	arr[2] = cubicInterpolate(p+width*2, y);
	arr[3] = cubicInterpolate(p+width*3, y);
	return cubicInterpolate(arr, x);
}

float *
make_buf ( int *ip, int width, int height )
{
	int *fp, i, j ;
	float *op, *tp, *ffp ;

	// printf("%s :: width %d height %d \n", __func__, width, height ) ;

	op = ( float * ) malloc (( width + WIDTH_ADJ ) * ( height + HEIGHT_ADJ ) * sizeof ( float )) ;

	if ( op == NULL )
	{
		printf("%s :: malloc fail w %d h %d \n", __func__, width, height ) ;
		return ( op ) ;
	}

	tp = op ;
	fp = ip ;
	i = width ;
	*tp++ = ( float )*fp ;
	while ( i-- )
	{
		*tp++ = ( float )*fp++ ;
	}
	fp-- ;
	// *tp++ = *fp ;	// WIDTH_ADJ back 2
	*tp++ = *fp++ ;

	fp = ip ;
	for ( i = 0 ; i < height ; i++ )
	{
		*tp++ = *fp ;
		for ( j = 0 ; j < width ; j++ )
		{
			*tp++ = *fp++ ;
		}
		fp-- ;
		// *tp++ = *fp ;
		*tp++ = *fp++ ;
	}

	ffp = tp - ( width + WIDTH_ADJ ) ;
	i = width + WIDTH_ADJ ;
	while ( i-- )
	{
		*tp++ = *ffp++ ;
	}

#ifdef CUDA_OBS 	// goog ... if ADJ is 2
	ffp = tp - ( width + WIDTH_ADJ ) ;
	i = width + WIDTH_ADJ ;
	while ( i-- )
	{
		*tp++ = *ffp++ ;
	}
#endif 

	// p_num_nm ("from", ip, width, height ) ;
	// p_num_nm_f ("done", op, width + WIDTH_ADJ, height + HEIGHT_ADJ ) ;

	return ( op );
}

float * 
do_resize( int *ip, int old_width, int old_height, int new_width, int new_height ) 
{
	int wid_adj, ii, jj, i, j ;
	float *tp, f, xf, yf, x_step, y_step ;
	float *output, *buftmp, x, y ;
#ifdef CUDA_OBS 
	unsigned int *uip ;

	x_step = ( float )(old_width - 1 ) / ( float )( new_width - 1 ) ;
	y_step = ( float )(old_height - 1 ) / ( float )( new_height - 1 ) ;
#endif 
	x_step = ( float )(old_width - 1 ) / ( float )( new_width ) ;
	y_step = ( float )(old_height - 1 ) / ( float )( new_height ) ;

	printf("%s : step width %f height %f ow %d %d w %d %d\n", __func__, x_step, y_step,
		 old_width, old_height, new_width, new_height ) ;

	if (( old_width != new_width ) || ( old_height != new_height ))
		buftmp = make_buf( ip, old_width, old_height ) ;
	else
	{
		i = old_width * old_height * sizeof ( int ) ;
		buftmp = ( float * )malloc ( i ) ;
		if ( buftmp == NULL )
		{
			printf("%s : buftmp malloc failed\n", __func__ ) ;
			return ( NULL ) ;
		} 
		memcpy ( buftmp, ip, i ) ;
		return (( float * )buftmp ) ;
	}

	if ( buftmp == NULL )
	{
		printf("%s : make_buf failed \n", __func__) ;
		return ( NULL ) ;
	}

	output = ( float * )malloc ( new_width * new_height * sizeof ( float )) ;

	if ( output == NULL )
	{
		printf("%s : output failed \n", __func__ ) ;
		free ( buftmp ) ;
		return ( NULL ) ;
	}

	wid_adj = old_width + WIDTH_ADJ ;

	tp = output ;
	xf = yf = 0 ;
	for ( i = 0 ; i < new_height ; i++ )
	{
		// printf("row %d -- %f \n", i, yf ) ;  
		for ( j = 0 ; j < new_width ; j++ )
		{
			ii = floorf( xf ) ;
			x = xf - ii ;

			jj = floorf ( yf ) ;
			y = yf - jj ;

			f = bicubicInterpolate ( buftmp + jj * wid_adj + ii, y, x, wid_adj ) ;

#ifdef CUDA_OBS 
			printf("col %d - xf %f - ii %d jj %d f %f x %f y %f\n", j, xf, ii, jj, f, x, y ) ;
			printf("--------------------------------------------------------------------\n") ;
#endif 

			*tp++ = f ;

			xf += x_step ;
		}
		xf = 0 ;
		yf += y_step ;
	}

	// p_num_nm_f ("done", output, new_width, new_height ) ;

	free ( buftmp ) ;

#ifdef CUDA_OBS 
	i = new_height * new_width ;
	uip = ( unsigned int * ) output ;
	tp = output ;
	while ( i-- )
		*uip++ = ( int ) ( *tp++ + 0.5 ) ;

	return (( int * )output ) ;
#endif 
	return ( output ) ;
}
