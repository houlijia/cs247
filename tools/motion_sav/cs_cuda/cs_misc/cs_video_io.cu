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
#include "cs_video_io.h"
#include "cs_config.h"
#include "cs_webcam.h"

#include "cs_motion_detect_v4.h"

#define CUDA_DBG 
// #define CUDA_ARROW_MISS

using namespace cv;
// const char *axis="http://PTSdisplay:ALUsurv@135.112.150.72:58006/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg" ;

static int sem_lock_init( int ) ;
#if 0
static void frame_plist( const char *s ) ;
static void sem_info( const char *s ) ;
#endif

static int frame_list_init ( int cnt, int gsize, int vsize, int bsize ) ;
static void * frame_grabber( void * ) ;
static void * frame_displayer( void *p) ;

static int sizeVideo, sizeGray ;
static int frame_per_io = 0 ;
static CvCapture *videoIn = NULL ;
static int videoIn_file;
static int to_display = 0 ;
static int video_input ;
static int htVideo, widVideo, ms_wait ;
static pthread_t grab_thread, display_thread ;
static int blk_cnt, blk_x = -1, blk_y = -1 ;
static int display_step_x = -1, display_step_y = -1 ;
static int md_x = -1, md_y = -1 ;
static int do_ignore_edge = 0 ;
static int disp_th_x = 0, disp_th_y = 0 ;
static int no_arrow = 0 ;
static float display_threshold = 0.0 ;

static sem_t sem_display, sem_ll, sem_filled, sem_free ;

static int displayGray = 1 ;

#define MIN_GRAY_SCALE	0
#define MAX_GRAY_SCALE	255
#define RGB_WHITE	255
#define RGB_BLACK	0
#define WHITE_THRESHOLD	192

// #define CUDA_OUTFILE
#ifdef CUDA_OUTFILE
static char *outputfile="/tmp/video.out" ;
// static char *outputfile="/home/ldl/mr/baotou_cs/test_data/video.out" ;
#endif 

#ifdef CUDA_OBS 
static void
phex( const char *s, char *cp, int cnt )
{
  static char phex_pbuf[128] ;
  static char map[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' } ;

	int pdone, i = 0 ;
	char *tp, c ;

	printf("%s: cp %p cnt %d\n", s, cp, cnt ) ;

	if ( !cnt )
	return ;

	tp = phex_pbuf ;
	while ( cnt > 0 )
	{
		pdone = 0 ;
		c = *cp++ ;
		cnt--;
		i++ ;

		// printf("--------------- i %d cnt %d c %x \n", i, cnt, c ) ;

		*tp++ = map[( c >> 4 ) & 0xf ] ;  
		*tp++ = map[ c & 0xf ] ;  

		if ( !( i % 32 ))
		{
			*tp = 0 ;

			printf("%s\n", phex_pbuf ) ;
			tp = phex_pbuf ;
			pdone++ ;
		} else if (!( i % 4 ))
		{
			*tp++ = ' ' ;
		}
	}

	if ( !pdone )
	{
		*tp = 0 ;
		printf("%s\n", phex_pbuf ) ;
	}
}
#endif	// #ifdef CUDA_OBS

// frame_rate is only used for VIDEO_SRC_FILE
int 
cs_vio_init( int frame_in_block, int width, int height, char *axis, int blkx, int blky,
	int md_xx, int md_yy, int display_threshold_x, int display_threshold_y, int video_source,
	int frame_rate, int do_display, int ignore_edge, int ignore_arrow, float t_display_threshold )
{
	int fps ;
	int i ;

	printf("%s: video source %d video src %s \n", __func__, video_source, axis ) ;

	to_display = do_display ;	// 1: to display 0: not
	no_arrow = ignore_arrow ;
	display_threshold = t_display_threshold ;

	video_input = video_source ;
	do_ignore_edge = ignore_edge ;

	switch ( video_source ) { 
	case VIDEO_SRC_IPCAM :

		videoIn = cvCaptureFromFile( axis );
		if ( videoIn == NULL )
		{
			printf("%s : Missing ipcam file %s\n", __func__, axis ) ;
			return ( 0 ) ;
		}
		printf("cs_vio_init: ipcam src %s \n", axis ) ;

		fps = (int) cvGetCaptureProperty(videoIn, CV_CAP_PROP_FPS);
		ms_wait = 80 ; // LDL .. wrong take out 

		break ;

	case VIDEO_SRC_WEBCAM :

		i = atoi ( axis ) ;
		videoIn = cvCaptureFromCAM( i );	// supposed to be 0, if only one webcam

		if ( videoIn == NULL )
		{
			printf("%s : Missing webcam file %s\n", __func__, axis ) ;
			return ( 0 ) ;
		}
		printf("cs_vio_init: webcam src %d \n", i ) ;

		fps = cs_webcam_fps( videoIn ) ;
		// cvSetCaptureProperty(videoIn, CV_CAP_PROP_FPS, 30);
		// fps = 30 ;
		// cvSetCaptureProperty(videoIn, CV_CAP_PROP_FPS, 30);
		// fps = cs_webcam_fps( videoIn ) ;
		ms_wait = 1000 / fps ;

		break;

	case VIDEO_SRC_FILE :
		
		if (( videoIn_file = open ( axis, O_RDONLY )) < 0 )
		{
			printf("%s : file %s errno %d \n", __func__, axis, errno ) ;
			return ( 0 ) ;
		}
		printf("%s: file src %s fid %d\n", __func__, axis, videoIn_file ) ;

		fps = frame_rate ;
		ms_wait = 1000 / fps ;

		break ;
	
	default :
		printf("%s: err video_source %d\n", __func__, video_source ) ;
		return ( 0 ) ;
	}

	printf("cs_vio_init: frame_in_blk %d w %d h %d x %d y %d ms_wait %d\n",
		frame_in_block, width, height, blkx, blky, ms_wait ) ;

	disp_th_x = display_threshold_x ;
	disp_th_y = display_threshold_y ;

	switch ( video_input ) {
	case VIDEO_SRC_WEBCAM :
	case VIDEO_SRC_IPCAM :

		htVideo = (int) cvGetCaptureProperty(videoIn, CV_CAP_PROP_FRAME_HEIGHT);
		widVideo = (int) cvGetCaptureProperty(videoIn, CV_CAP_PROP_FRAME_WIDTH);

		if (( width != widVideo ) || ( height != htVideo ))
		{
			printf("vio_init: mismatch wid %d %d height %d %d\n",
				width, widVideo, height, htVideo ) ;
			return ( 0 ) ;
		}

		sizeVideo = htVideo * widVideo * 3 ; // for max ... but should set it to media specific
		sizeGray = htVideo * widVideo ;

		break ;

	case VIDEO_SRC_FILE :

		widVideo = width ;
		htVideo = height ;

		sizeVideo = htVideo * widVideo * 2 ; // for YUV422
		sizeGray = htVideo * widVideo ;

		break ;
	}

#ifndef OPENCV_3
	if ( do_display )
		cvNamedWindow("Display Camera PPP", CV_WINDOW_AUTOSIZE ) ;
#endif 
#ifdef CUDA_OBS 
	cvNamedWindow("Display Camera CCC", CV_WINDOW_AUTOSIZE ) ;
	cvNamedWindow("Display Camera GRAY", CV_WINDOW_AUTOSIZE ) ;
#endif 

	if (!sem_lock_init( do_display ))
		return ( 0 ) ;

	// sem_info("after init") ;

	blk_x = blkx ;
	blk_y = blky ;
	blk_cnt = blk_x * blk_y ;

	display_step_x = width / ( blk_x - 1 ) ;
	display_step_y = height / ( blk_y - 1 ) ;

	md_x = md_xx ;
	md_y = md_yy ;

	printf ("vio_init: Video Size: width = %d, height = %d, fps = %d wait %d"
		" stepx %d stepy %d blkx %d blky %d mdx %d mdy %d disp %d %d\n",
		widVideo, htVideo, fps, ms_wait, display_step_x, display_step_y, blk_x,
		blk_y, md_x, md_y, disp_th_x, disp_th_y );

	if ( !frame_list_init ( frame_in_block, sizeGray, sizeVideo, blk_cnt ))
		return ( 0 ) ;

	// sem_info("after frame_init") ;

	// frame_plist("after frame_init") ;

	pthread_create( &grab_thread, NULL, frame_grabber, NULL ) ;

	if ( to_display )
		pthread_create( &display_thread, NULL, frame_displayer, NULL ) ;

	return ( 1 ) ;
}

#if 0
// sem support 

static void 
sem_info( const char *s )
{
	int j, i, k, l ;

	if (( sem_getvalue( &sem_ll, &i )) < 0 )
		printf("sem_ll: info failed %d\n", errno ) ;

	if (( sem_getvalue( &sem_filled, &j )) < 0 )
		printf("sem_filled: info failed %d\n", errno ) ;

	if (( sem_getvalue( &sem_free, &k )) < 0 )
		printf("sem_free: info failed %d\n", errno ) ;

	if ( to_display )
		if (( sem_getvalue( &sem_display, &l )) < 0 )
			printf("sem_display: info failed %d\n", errno ) ;

	printf("sem_info: %s: ll %d filled %d free %d display %d\n", s, i, j, k, l ) ;
}
#endif	// #if 0

static int
sem_lock_init( int do_display )
{
	if ( sem_init( &sem_ll, 0, 1 ))
	{
		printf("sem_ll: failed %d\n", errno ) ;
		return ( 0 ) ;
	}

	if ( sem_init( &sem_filled, 0, 0 ))
	{
		printf("sem_filled: failed %d\n", errno ) ;
		return ( 0 ) ;
	}

	if ( sem_init( &sem_free, 0, 0 ))
	{
		printf("sem_free: failed %d\n", errno ) ;
		return ( 0 ) ;
	}

	if ( do_display )
	{
		if ( sem_init( &sem_display, 0, 0 ))
		{
			printf("sem_display: failed %d\n", errno ) ;
			return ( 0 ) ;
		}
	}
	return ( 1 ) ;
}

// linked list ...

struct frame_list_head {
	struct frame_list *first ;
	struct frame_list *last ;
} ;

static struct frame_list_head free_frame_head ;
static struct frame_list_head filled_frame_head ;
static struct frame_list_head display_frame_head ;

#define NUM_FRAME_LIST_HEAD		3
#define NUM_FRAME_LIST		6

#if 0
static void
frame_plist( const char *s )
{
	struct frame_list *fp ;
	int i ;
	struct frame_list_head *flhp ;

#ifdef CUDA_DBG 
	printf("-------------------------------------------- frame_plist: %s\n", s ) ;
#endif 

	sem_wait( &sem_ll ) ;

	for ( i = 0 ; i < NUM_FRAME_LIST_HEAD ; i++ )
	{
		flhp = NULL ;
		switch ( i ) {
		case 0 :
			flhp = &free_frame_head ;
		
#ifdef CUDA_DBG 
			printf("frame_plist: free : ") ;
#endif 
			break ;

		case 1 :	
			flhp = &filled_frame_head ;
#ifdef CUDA_DBG 
			printf("frame_plist: filled : ") ;
#endif 
			break ;

		default :
			if ( to_display )
			{
				flhp = &display_frame_head ;
#ifdef CUDA_DBG 
				printf("frame_plist: display : ") ;
#endif 
			}
			break ;
		}

#ifdef CUDA_DBG 

		if ( flhp )
		{
			printf("hp %p first %p last %p \n", flhp, flhp->first,
				flhp->last ) ;

			fp = flhp->first ;
			while ( fp )
			{
				printf("	fp %p \n", fp ) ;
				fp = fp->np ;
			}
		}
#endif 
	}
	sem_post ( &sem_ll ) ;

#ifdef CUDA_DBG 
	printf("END ------------------------------------------- frame_plist: %s\n", s ) ;
#endif 
}
#endif	// #if 0

static void
frame_put( struct frame_list_head *flhp, frame_list *fp  )
{
#ifdef CUDA_OBS 
	if ( flhp == &free_frame_head )
	{
		printf("frame_put: ----- free: ") ;
	} else if ( flhp == &filled_frame_head ) 
	{
		printf("frame_put: ----- filled: ") ;
	} else
	{
		printf("frame_put: ----- display: ") ;
	}

	printf(" flhp %p fp %p\n", flhp, fp ) ;
#endif 

	sem_wait( &sem_ll ) ;

	if ( flhp->last != NULL )
	{
		flhp->last->np = fp ;
		flhp->last = fp ;
	} else
		flhp->last = flhp->first = fp ;

	fp->np = NULL ;

	sem_post( &sem_ll ) ;

	if ( flhp == &free_frame_head )
		sem_post( &sem_free ) ;
	else if ( flhp == &filled_frame_head ) 
		sem_post( &sem_filled ) ;
	else
		sem_post( &sem_display ) ;

	// frame_plist("after frame_put") ;

	// sem_info("------------------- after put one") ;

}

static struct frame_list *
frame_get ( struct frame_list_head *flhp )
{
	struct frame_list *fp ;
#ifdef CUDA_OBS 
	int qtype ;
#endif 

	while ( 1 )
	{
		if ( flhp == &free_frame_head )
		{
#ifdef CUDA_OBS 
			printf("frame_get: free: " ) ;
			qtype = 1 ;
#endif 
			sem_wait( &sem_free ) ;
		} else if ( flhp == &filled_frame_head )
		{
#ifdef CUDA_OBS 
			printf("frame_get: filled: " ) ;
			qtype = 2 ;
#endif 
			sem_wait( &sem_filled ) ;
		} else
		{
#ifdef CUDA_OBS 
			printf("frame_get: display: " ) ;
			qtype = 3 ;
#endif 
			sem_wait( &sem_display ) ;
		}

		// sem_info("frame_get") ;

		// frame_plist("frame_get: before") ;

		sem_wait( &sem_ll ) ;

		fp = flhp->first ;

#ifdef CUDA_OBS 
		printf("fp %p\n", fp ) ;
#endif 

		if ( fp )
		{
			flhp->first = fp->np ;

			if ( flhp->first == NULL )
				flhp->last = NULL ;

			fp->np = NULL ;

			sem_post( &sem_ll ) ;

			// sem_info("--------------- after got one") ;

			// frame_plist("frame_get: after frame_get") ; 

#ifdef CUDA_OBS 
			printf("frame_get: ========= type %d got %p\n", qtype, fp ) ;
#endif 

			return ( fp ) ;

		} 
		sem_post( &sem_ll ) ;

	}
}

// cnt is the number of buffers 

static int 
frame_list_init ( int cnt, int gsize, int vsize, int bsize )
{
	struct frame_list *fp ;
	int i ;

#ifdef CUDA_OBS 
	printf("%s :: cnt %d gsize %d vsize %d bsize %d \n",
		__func__, cnt, gsize, vsize, bsize ) ;
#endif 

	frame_per_io = cnt ;

	free_frame_head.first = free_frame_head.last = NULL ;
	filled_frame_head.first = filled_frame_head.last = NULL ;
	display_frame_head.first = display_frame_head.last = NULL ;

	fp = ( struct frame_list * ) malloc ( sizeof ( *fp ) * NUM_FRAME_LIST ) ;

	for ( i = 0 ; i < NUM_FRAME_LIST ; i++ )
	{
		fp->vbp = ( char * ) malloc ( vsize * frame_per_io ) ;
		fp->gbp = ( char * ) malloc ( gsize * frame_per_io ) ;
		fp->outp = ( int * ) malloc ( bsize * sizeof( int ) * OUTP_ENTRY_SIZE ) ;

		printf("frame_list_init: i %d fp %p vbp %p gbp %p outp %p\n", i, fp,
			fp->vbp, fp->gbp, fp->outp ) ;

		if ( !fp->vbp || !fp->gbp || !fp->outp )
		{
			printf("frame_list_init: malloc failed \n") ;
			return ( 0 ) ;
		}

		fp->np = NULL ;

		frame_put ( &free_frame_head, fp ) ;

		fp++ ;
	}
	return ( 1 ) ;
}

// child process ... take off free_frame put to filled_frame

static void
do_frame_grabber( void * )
{
	struct frame_list *fp ;
	IplImage *current_frame = NULL, *gray_frame = NULL ;
	int i, file_done, frame_cnt = 0, new_frame = 1 ;

	gray_frame = cvCreateImage(cvSize(widVideo,htVideo),IPL_DEPTH_8U,1);

	sem_wait( &sem_filled ) ;	// block here wait for the reader to be ready 

	file_done = 0 ;

	while ( 1 )
	{
		if ( new_frame )
		{
			fp = frame_get ( &free_frame_head ) ;

#ifdef CUDA_OBS 
			printf("grabber *** new fp %p vbp %p gbp %p \n", fp, fp->vbp, fp->gbp ) ;
#endif 

			fp->cnt = 0 ;
			new_frame = 0 ;
		}

		switch ( video_input ) {
		case VIDEO_SRC_IPCAM :
		case VIDEO_SRC_WEBCAM :

			current_frame = cvQueryFrame( videoIn ) ;	
			frame_cnt++ ;

#ifdef CUDA_OBS 
			cvShowImage( "Display Camera CCC", current_frame ) ;

			cvWaitKey( 10 ) ;
#endif 

			memcpy ( fp->vbp + sizeVideo * fp->cnt, current_frame->imageData,
				sizeVideo ) ;

			printf("=== size %d === \n", current_frame->imageSize ) ;

			cvCvtColor( current_frame, gray_frame, CV_BGR2GRAY ) ;

#ifdef CUDA_OBS 
			cvShowImage( "Display Camera GRAY", gray_frame ) ;

			cvWaitKey( 10 ) ;
#endif 
#ifdef CUDA_OBS 
			phex("color", current_frame->imageData, 192 ) ;
			phex("gray", gray_frame->imageData, 192 ) ;
#endif 

#ifdef CUDA_OBS 
			printf("=== fr %d : size %d data %p wid %d high %d n_chan %d dep %d ws %d\n"
				"order %d orig %d roi %p\n",
				frame_cnt,
				gray_frame->imageSize,
				gray_frame->imageData,
				gray_frame->width,
				gray_frame->height,
				gray_frame->nChannels,
				gray_frame->depth,
				gray_frame->widthStep,

				gray_frame->dataOrder,
				gray_frame->origin,
				gray_frame->roi
			) ;
#endif 

			memcpy ( fp->gbp + sizeGray * fp->cnt, gray_frame->imageData,
				sizeGray ) ;

			break ;

		case VIDEO_SRC_FILE :
			// assuming it is YUV422 as in pets_test_enc ... should add more flavors

#ifdef CUDA_OBS 
			printf("%s ::: read file %d \n", __func__, videoIn_file ) ;
#endif 

			if ( file_done )
				break ;
			
		  	if (( i = read ( videoIn_file,  fp->vbp + sizeVideo * fp->cnt, sizeVideo )) != sizeVideo )
			{
				printf("%s: file reading error got %d frame_cnt %d \n", __func__, i, frame_cnt ) ;
				fp->cnt = -1 ;	 // the following + will make it 0 ... signal the EOF to caller 
				file_done++ ;
			} else
			{
				frame_cnt++ ;
				memcpy ( fp->gbp + sizeGray * fp->cnt, fp->vbp + sizeVideo * fp->cnt, sizeGray ) ;
			}
		}

		fp->cnt++ ;

		// sleep( 1 ) ;

		if (( file_done ) || ( fp->cnt == frame_per_io ))
		{
			new_frame++ ;
			frame_put ( &filled_frame_head, fp ) ; 
#ifdef CUDA_OBS 
			printf("grabber: put to filled %p\n", fp) ;
#endif 
		}
	}

}

static void *
frame_grabber( void * ptr)
{
  do_frame_grabber(ptr);
  return NULL;
}

struct frame_list *
cs_vio_get()
{
	struct frame_list *fp ;

	fp = frame_get( &filled_frame_head ) ;

#ifdef CUDA_OBS 
	printf("cs_vio_get *** fp %p vbp %p gbp %p \n", fp, fp->vbp, fp->gbp ) ;
#endif 

	return ( fp ) ;
}


#define MAX_ARROW_MAG		20
#define MIN_ARROW_MAG		5	

static void
drawArrow(IplImage *image, CvPoint from, CvPoint to, CvScalar color,
        float arrow_ratio, int thickness=1, int line_type=8, int shift=0)
{
	int arrowMagnitude ;

#ifdef CUDA_OBS 
	printf("%s ::: arrow_ratio %f \n", __func__, arrow_ratio ) ;
#endif 

	arrowMagnitude = arrow_ratio * ( MAX_ARROW_MAG - MIN_ARROW_MAG ) + MIN_ARROW_MAG ;

	//Draw the principle line
	cvLine(image, from, to, color, thickness, line_type, shift);
	const double CS_PI = 3.141592653 ;

	//compute the angle alpha
	double angle = atan2((double)from.y-to.y, (double)from.x-to.x);

	//compute the coordinates of the first segment
	from.x = (int) ( to.x +  arrowMagnitude * cos(angle + (CS_PI/8)));
	from.y = (int) ( to.y +  arrowMagnitude * sin(angle + (CS_PI/8)));

	//Draw the first segment
	cvLine(image, from, to, color, thickness, line_type, shift);

	//compute the coordinates of the second segment
	from.x = (int) ( to.x +  arrowMagnitude * cos(angle - (CS_PI/8)));
	from.y = (int) ( to.y +  arrowMagnitude * sin(angle - (CS_PI/8)));

	//Draw the second segment
	cvLine(image, from, to, color, thickness, line_type, shift);
}  


#ifdef CUDA_OBS
static void
drawArrow(IplImage *image, CvPoint from, CvPoint to, CvScalar color,
        int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0)
{
	//Draw the principle line
	cvLine(image, from, to, color, thickness, line_type, shift);
	const double CS_PI = 3.141592653 ;

	//compute the angle alpha
	double angle = atan2((double)from.y-to.y, (double)from.x-to.x);

	//compute the coordinates of the first segment
	from.x = (int) ( to.x +  arrowMagnitude * cos(angle + (CS_PI/8)));
	from.y = (int) ( to.y +  arrowMagnitude * sin(angle + (CS_PI/8)));

	//Draw the first segment
	cvLine(image, from, to, color, thickness, line_type, shift);

	//compute the coordinates of the second segment
	from.x = (int) ( to.x +  arrowMagnitude * cos(angle - (CS_PI/8)));
	from.y = (int) ( to.y +  arrowMagnitude * sin(angle - (CS_PI/8)));

	//Draw the second segment
	cvLine(image, from, to, color, thickness, line_type, shift);
}  
#endif	// #ifdef CUDA_OBS

void
cs_vio_put( struct frame_list *fp )
{
#ifdef CUDA_OBS 
	int *dp, i, j, h, v, t, oh, ov, ot ;
	float ova, va, *ffp ;
#endif 

	printf("cs_vio_put: fp %p outp %p ----------------------------------------\n",
		fp, fp->outp ) ;

#ifdef CUDA_OBS 
	dp = fp->outp ;
	for ( i = 0 ; i < blk_y ; i++ )
	{
		for ( j = 0 ; j < blk_x ; j++ )
		{
			t = *dp++ ;
			v = *dp++ ;
			h = *dp++ ;
			ffp = ( float * )dp++ ;
			va = *ffp ;

			ot = *dp++ ;
			ov = *dp++ ;
			oh = *dp++ ;
			ffp = ( float * )dp++ ;
			ova = *ffp ;

			printf("BLK %d %d : t %d v %d h %d va %f, ot %d ov %d oh %d ova %f\n",
				j, i, t, v, h, va, ot, ov, oh, ova ) ;
		}
	}

	printf("END ================================================================\n") ;
#endif 
	
	// the pic based on what is in outp first ... then return the whole frame_list

	if ( to_display )
		frame_put( &display_frame_head, fp ) ;
	else
		frame_put( &free_frame_head, fp ) ;
}

void
cs_vio_start()
{
#ifdef CUDA_OBS 
	printf("%s ::: is called \n", __func__ );
#endif 
	sem_post( &sem_filled ) ;	// ok ... free the grabber ...
}

// fp points to the motion detection data ... we do this for drawing the motion arrows
// in each block
void
cs_vio_record ( int *fp, int *tp ) 
{
	memcpy ( tp, fp, blk_cnt * OUTP_ENTRY_SIZE * sizeof ( int )) ;
}

// take off display_frame_head, put to free_frame for testing purpose

static void *
frame_displayer( void *p)
{
	struct frame_list *fp ;
	IplImage *current_frame = 0 ;
	int oxpos, oypos, xpos, ypos, frame_cnt = 0 ;
	int *dp, i, j, k, h, v ;
#if !defined(OPENCV_3) && defined(CUDA_OBS)
	int t ;
	int gray_scale;
#endif
	float ftmp, *ffp ;
	float va, va_orig, va_min ;
	float gray_ratio, va_total, ratio ;
	int va_cnt ;
#ifdef CUDA_ARROW_MISS 
	int arrow_cnt ;
	int miss_cnt ;
#endif 

#ifdef CUDA_OUTFILE
	int ofid ;

	if ((ofid = open ( outputfile, O_WRONLY | O_CREAT | O_TRUNC,  S_IRWXU )) < 0 ) 
	{
		printf("%s : open file %s failed errno %d \n", __func__, outputfile, errno ) ;
		return ( NULL ) ;
	}

	printf("open %s fid $d \n", outputfile, ofid ) ;
#endif 

	// cvNamedWindow("Display Camera PPP", CV_WINDOW_AUTOSIZE ) ;

#ifdef OPENCV_3
	namedWindow( "Display window", WINDOW_AUTOSIZE );
#endif 

	if ( !to_display )
	{
		printf("%s : error to_display %d \n", __func__, to_display ) ;
		return ( NULL ) ;
	}

	if ( displayGray )
		current_frame = cvCreateImage(cvSize(widVideo,htVideo),IPL_DEPTH_8U,1);
	else
		current_frame = cvCreateImage(cvSize(widVideo,htVideo),IPL_DEPTH_8U,3);

	while ( 1 )
	{
		fp = frame_get ( &display_frame_head ) ;

#ifdef CUDA_OBS 
		printf("frame_displayer *** fp %p vbp %p gbp %p outp %p\n", fp, fp->vbp, fp->gbp,
			 fp->outp ) ;
#endif 

#ifdef CUDA_ARROW_MISS
		arrow_cnt = 0 ;
		miss_cnt = 0 ;
#endif 

		for ( k = 0 ; k < frame_per_io ; k++ )
		{
			if ( displayGray )
				current_frame->imageData = fp->gbp + sizeGray * k ;
			else
				current_frame->imageData = fp->vbp + sizeVideo * k ;

#ifdef CUDA_OBS 
			printf("frame_displayer === fp %p vbp %p gbp %p k %d outp %p\n",
				fp, fp->vbp, fp->gbp, k, fp->outp ) ;
#endif 

			if ( !no_arrow )
			{
				if ( !k )
				{
					va = 0.0 ;	// max
					va_min = 0 ;
					dp = fp->outp ;
					j = blk_cnt ;
					va_total = 0 ;
					va_cnt = 0 ;

					for ( i = 0; i < j ; i++ )
					{
#if !defined(OPENCV_3) && defined(CUDA_OBS)
						t = *dp++ ;
#else
						dp++;
#endif
						v = *dp++ ;
						h = *dp++ ;

						ffp = ( float *) dp++ ;
						ftmp = *ffp ;

						dp += L1_NORM_STEP4_RETURN_ENTRY_SIZE ; 

						if (( ftmp > 0 ) && (( abs ( h - md_x ) > disp_th_x) ||
							( abs( v - md_y ) > disp_th_y )))
						{
							// for 'mean' method
							va_total += ftmp ;
							va_cnt++ ;

							// for min-max method
							if ( ftmp > va )
								va = ftmp ;			// max

							if ( ftmp < va_min )
								va_min = ftmp ;	// min	
						}
					}

					// for min-max
					ratio = 1.0 / ( va - va_min ) ;
					// ratio = 1.0 / va ;	// ignore the min ... va_orig for now

#ifdef CUDA_OBS 
					printf("%s ::: total %f cnt %d ratio %f va %f va_min %f total %f\n",
						__func__, va_total, va_cnt, ratio, va, va_min, va_total/va_cnt ) ;
#endif 

					// for 'mean'
					va_total /= va_cnt ;

					// should save the result for the next 8 frames ... QQQ
				}

				dp = fp->outp ;
				ypos = 0 ;

				for ( i = 0 ; i < blk_y ; i++ )
				{
					xpos = 0 ;
					for ( j = 0 ; j < blk_x ; j++ )
					{
#if !defined(OPENCV_3) && defined(CUDA_OBS)
						t = *dp++ ;
#else
						dp++;
#endif
						v = *dp++ ;
						h = *dp++ ;

						ffp = ( float *) dp ;
						va = *ffp ;

						dp += L1_NORM_STEP4_RETURN_ENTRY_SIZE ;	// should have a better name ...
						ffp = ( float *) dp++ ;
						va_orig = *ffp ;

#ifdef CUDA_OBS 
						printf("RAW  k %d i %d j %d va %f va_orig %f \n", k, i, j, va, va_orig ) ;
#endif 

#ifdef CUDA_OBS 
						// for min-max
						gray_scale = RGB_WHITE - ((( ratio * ( va - va_min )) * 
							(( float ) MAX_GRAY_SCALE - ( float )MIN_GRAY_SCALE ))
							+ ( float )MIN_GRAY_SCALE ) ;

						printf("GRAY %d ", gray_scale ) ;

						if ( gray_scale < WHITE_THRESHOLD )
							gray_scale = RGB_BLACK ;
						else
							gray_scale = RGB_WHITE ;
#endif 

#ifdef CUDA_OBS 
						// for 'mean'
						if ( va_total < va ) 	// better than average
							gray_scale = RGB_BLACK ;
						else
							gray_scale = RGB_WHITE ;
#endif 

#ifdef CUDA_OBS 
						printf(" --- va %f total %f gray_scale %d \n", va, va_total, gray_scale ) ;
#endif 

						if (( va > 0 ) &&
							(( abs ( h - md_x ) > disp_th_x) || ( abs( v - md_y ) > disp_th_y )))
						{

							// if va < 0, then va_orig is < 0 too ... so every bet is off.
#ifdef CUDA_OBS 
							printf("%s: frame_cnt %d i %d j %d t %d v %d h %d va %f x/y %d %d\n",
								__func__, frame_cnt, i, j, t, v, h, va, xpos, ypos ) ;

							printf("k %d i %d j %d va %f va_orig %f ratio %f ratio %f gray %d\n",
								k, i, j, va, va_orig, va/va_orig, ratio, gray_scale ) ;
#endif 
							gray_ratio = ( va - va_min ) * ratio ;

							va = ( va - va_orig ) / va ;

#ifdef CUDA_OBS 
							printf("	final va %f \n", va ) ;
#endif 

#ifdef CUDA_OBS 
							if (( abs ( va ) < display_threshold ) && !k )
							{
								miss_cnt++ ;
								printf(" --- misscnt %d va %f display_threshold %f i/j/k %d %d %d\n",
									miss_cnt, va, display_threshold, i, j, k ) ;
							}
#endif 

							if (( abs ( va ) > display_threshold ) &&
								(( !do_ignore_edge ) || !( do_ignore_edge && (
								(( i == 0 ) || ( i == ( blk_y - 1 )) || 
								( j == 0 ) || ( j == ( blk_x - 1 )))))))
							{

								if (!(( i == 0 ) || ( i == ( blk_y - 1 )) || 
									( j == 0 ) || ( j == ( blk_x - 1 ))))
								{
									oxpos = xpos + ((( k - frame_per_io / 2 ) * ( h - md_x )) * 6 ) / frame_per_io ; 
									oypos = ypos + ((( k - frame_per_io / 2 ) * ( v - md_y )) * 6 ) / frame_per_io ;
								} else {
									oxpos = xpos ; 
									oypos = ypos ;
								}

								// use the arrow
								drawArrow( current_frame, cvPoint(oxpos,oypos),
									cvPoint( oxpos + ( h - md_x ) * 6, oypos + ( v - md_y ) * 6),
									CV_RGB(0, 0, 0),
									gray_ratio ) ;
#ifdef CUDA_OBS 
								// use the gray scale
								drawArrow( current_frame, cvPoint(oxpos,oypos),
									cvPoint( oxpos + ( h - md_x ) * 6, oypos + ( v - md_y ) * 6),
									CV_RGB(gray_scale, gray_scale, gray_scale )) ;
									// CV_RGB(0,0,0)) ;
#endif 

#ifdef CUDA_ARROW_MISS 
								if (!k )
								{
									printf("%s :: arrow t %d v %d h %d\n", __func__,
										t, v, h ) ;	
									arrow_cnt++ ;
								}
#endif 
							}
						}

						xpos += display_step_x ;
					}
					ypos += display_step_y ;
				}
			}

#ifndef OPENCV_3
			cvShowImage( "Display Camera PPP", current_frame ) ;
			cvWaitKey( ms_wait ) ;
#else
			cv::Mat image1 = cv::cvarrToMat(current_frame) ;
			imshow("Display window", image1 ) ;
			waitKey( ms_wait ) ;
#endif 

#ifdef CUDA_OUTFILE
			i = write( ofid, current_frame->imageData, sizeGray ) ;

			if ( i != sizeGray )
				printf("	ofid write fail %d errno %d size %d \n", i, errno, sizeGray ) ;
#endif 

			frame_cnt++ ;

		}

		frame_put ( &free_frame_head, fp ) ;

#ifdef CUDA_ARROW_MISS 
		printf("\n%s: cnt %d arrow %d miss_cnt %d\n", __func__, frame_cnt, arrow_cnt, miss_cnt ) ;
#endif 
	}
}
