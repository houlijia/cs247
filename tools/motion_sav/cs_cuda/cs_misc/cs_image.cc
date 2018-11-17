#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <semaphore.h>
#include <pthread.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include "cs_image.h"

IplImage *current_frame = NULL ;

int 
cs_image_init( int col, int row, int num_of_channels ) 
{
	cvNamedWindow("Camera Image", CV_WINDOW_AUTOSIZE ) ;

	if ( col % 4 )
	{
		printf("%s : buf err col %d is not mod 4\n", __func__, col ) ;
		return ( 0 ) ;
	}

	current_frame = cvCreateImageHeader( cvSize( col, row ), IPL_DEPTH_8U, num_of_channels );

	if ( current_frame == NULL )
	{
		printf("%s : current frame failed \n", __func__ ) ;
		return (0 ) ;
	}

	current_frame->widthStep = col * num_of_channels ;
	current_frame->width = col ;
	current_frame->height = row ;
	current_frame->imageSize = col * num_of_channels * row ;

	current_frame->nChannels = num_of_channels ;
	current_frame->depth = IPL_DEPTH_8U ;

	current_frame->maskROI = NULL ;
	current_frame->imageId = NULL ;
	current_frame->tileInfo = NULL ;
	current_frame->imageDataOrigin = NULL ; // ?
	current_frame->dataOrder = 0 ; // interleaved color channels

	return ( 1 ) ;
}

void
cs_image_show ( char *bgrp )
{
	current_frame->imageData = bgrp ;

	cvShowImage( "Camera Image", current_frame ) ;

	cvWaitKey( 300 ) ;
}
