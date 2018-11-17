#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <sys/time.h>
#include <opencv2/opencv.hpp>

#include "cs_webcam.h"

#define FRAME_FLUSH_CNT 30
#define FRAME_CNT 100

int
cs_webcam_fps( CvCapture *videoIn )
{
#if 0
	IplImage *fp ;
#endif
	struct timeval tv, ttv ;
	int i ;
	float fi ;

	i = FRAME_FLUSH_CNT ;
	while ( i-- )
#if 0
		fp =
#endif
		  cvQueryFrame( videoIn ) ;

	if ( gettimeofday( &tv, NULL ))
		return ( 0 ) ;

	i = FRAME_CNT ;
	while ( i-- )
#if 0
		fp = 
#endif
		  cvQueryFrame( videoIn ) ;

	if ( gettimeofday( &ttv, NULL ))
		return ( 0 ) ;

	printf("%s: ttv %ld %ld tv %ld %ld \n", __func__,
	       long(ttv.tv_sec), long(ttv.tv_usec), long(tv.tv_sec), long(tv.tv_usec) ) ;

	i = ( ttv.tv_sec - tv.tv_sec ) * 1000000 + ttv.tv_usec - tv.tv_usec ;

	printf("%s: i is %d \n", __func__, i ) ;

	fi = i ;

	fi = ( 1000000 * FRAME_CNT ) / fi ;

	i = round ( fi ) ;

	printf("%s: fps is %f %d \n", __func__, fi, i ) ;

	return ( i ) ;
}
