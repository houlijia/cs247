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
#include "cs_ipcam.h"
#include "cs_config.h"
#include "cs_webcam.h"

#define CUDA_DBG 

using namespace cv;
// const char *axis="http://PTSdisplay:ALUsurv@135.112.150.72:58006/axis-cgi/mjpg/video.cgi?resolution=640x480&req_fps=30&.mjpg" ;


static int sem_lock_init() ;
#if 0
static void frame_plist( const char *s ) ;
static void sem_info( const char *s ) ;
#endif
static int frame_list_init ( int cnt, int gsize, int vsize, int bsize ) ;
static void * frame_grabber( void * ) ;
static void * frame_displayer( void *p) ;

static int sizeVideo, sizeGray ;
static int frame_per_io = 0 ;
static CvCapture *videoIn ;
static int htVideo, widVideo, ms_wait ;
static pthread_t grab_thread, display_thread ;
static int blk_x = -1, blk_y = -1 ;
static int display_step_x = -1, display_step_y = -1 ;
static int md_x = -1, md_y = -1 ;
static int disp_th_x = 0, disp_th_y = 0 ;

static sem_t sem_display, sem_ll, sem_filled, sem_free ;

static int displayGray = 1 ;

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

int 
cs_ipcam_init( int frame_in_block, int width, int height, char *axis, int blkx, int blky,
	int md_xx, int md_yy, int display_threshold_x, int display_threshold_y, int video_source )
{
	int fps ;
	int i ;

	printf("%s: video source %d video src %s \n", __func__, video_source, axis ) ;

	if ( video_source == VIDEO_SRC_IPCAM )
	{
		videoIn = cvCaptureFromFile( axis );
		printf("cs_ipcam_init: ipcam src %s \n", axis ) ;
	} else 
	{
		i = atoi ( axis ) ;
		videoIn = cvCaptureFromCAM( 0 );
		printf("cs_ipcam_init: webcam src %d \n", i ) ;
	}

	printf("cs_ipcam_init: frame_in_blk %d w %d h %d x %d y %d \n",
		frame_in_block, width, height, blkx, blky ) ;

	disp_th_x = display_threshold_x ;
	disp_th_y = display_threshold_y ;

	if ( videoIn == NULL )
	{
		printf("ipcam_init: Missing file %s\n", axis ) ;
		return ( 0 ) ;
	}

	htVideo = (int) cvGetCaptureProperty(videoIn, CV_CAP_PROP_FRAME_HEIGHT);
	widVideo = (int) cvGetCaptureProperty(videoIn, CV_CAP_PROP_FRAME_WIDTH);

	if (( width != widVideo ) || ( height != htVideo ))
	{
		printf("ipcam_init: mismatch wid %d %d height %d %d\n",
			width, widVideo, height, htVideo ) ;
		return ( 0 ) ;
	}

	sizeVideo = htVideo * widVideo * 3 ;
	sizeGray = htVideo * widVideo ;

	if ( video_source == VIDEO_SRC_IPCAM )
	{
		fps = (int) cvGetCaptureProperty(videoIn, CV_CAP_PROP_FPS);
		ms_wait = 80 ; // LDL .. wrong take out 
	} else 
	{
		fps = cs_webcam_fps( videoIn ) ;
		// cvSetCaptureProperty(videoIn, CV_CAP_PROP_FPS, 30);
		// fps = 30 ;
		// cvSetCaptureProperty(videoIn, CV_CAP_PROP_FPS, 30);
		// fps = cs_webcam_fps( videoIn ) ;
		ms_wait = 1000 / fps ;
	}

	cvNamedWindow("Display Camera PPP", CV_WINDOW_AUTOSIZE ) ;
#ifdef CUDA_OBS 
	cvNamedWindow("Display Camera CCC", CV_WINDOW_AUTOSIZE ) ;
	cvNamedWindow("Display Camera GRAY", CV_WINDOW_AUTOSIZE ) ;
#endif 

	if (!sem_lock_init())
		return ( 0 ) ;

	// sem_info("after init") ;

	blk_x = blkx ;
	blk_y = blky ;

	display_step_x = width / ( blk_x - 1 ) ;
	display_step_y = height / ( blk_y - 1 ) ;

	md_x = md_xx ;
	md_y = md_yy ;

	printf ("ipcam_init: Video Size: width = %d, height = %d, fps = %d wait %d"
		" stepx %d stepy %d blkx %d blky %d mdx %d mdy %d disp %d %d\n",
		widVideo, htVideo, fps, ms_wait, display_step_x, display_step_y, blk_x,
		blk_y, md_x, md_y, disp_th_x, disp_th_y );

	if ( !frame_list_init ( frame_in_block, sizeGray, sizeVideo, blk_x * blk_y ))
		return ( 0 ) ;

	// sem_info("after frame_init") ;

	// frame_plist("after frame_init") ;

	pthread_create( &grab_thread, NULL, frame_grabber, NULL ) ;
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

	if (( sem_getvalue( &sem_display, &l )) < 0 )
		printf("sem_display: info failed %d\n", errno ) ;

	printf("sem_info: %s: ll %d filled %d free %d display %d\n", s, i, j, k, l ) ;
}
#endif				// #if 0

static int
sem_lock_init()
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

	if ( sem_init( &sem_display, 0, 0 ))
	{
		printf("sem_display: failed %d\n", errno ) ;
		return ( 0 ) ;
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
			flhp = &display_frame_head ;
#ifdef CUDA_DBG 
			printf("frame_plist: display : ") ;
#endif 
			break ;
		}

#ifdef CUDA_DBG 
		printf("hp %p first %p last %p \n", flhp, flhp->first,
			flhp->last ) ;

		fp = flhp->first ;
		while ( fp )
		{
			printf("	fp %p \n", fp ) ;
			fp = fp->np ;
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
	//	int qtype ;

	while ( 1 )
	{
		if ( flhp == &free_frame_head )
		{
#ifdef CUDA_OBS 
			printf("frame_get: free: " ) ;
#endif 
			// qtype = 1 ;
			sem_wait( &sem_free ) ;
		} else if ( flhp == &filled_frame_head )
		{
#ifdef CUDA_OBS 
			printf("frame_get: filled: " ) ;
#endif 
			// qtype = 2 ;
			sem_wait( &sem_filled ) ;
		} else
		{
#ifdef CUDA_OBS 
			printf("frame_get: display: " ) ;
#endif 
			// qtype = 3 ;
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

			// printf("frame_get: ========= type %d got %p\n", qtype, fp ) ;

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

		printf("frame_list_init: i %d vbp %p gbp %p\n", i, fp->vbp, fp->gbp ) ;

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
	int frame_cnt = 0, new_frame = 1 ;

	gray_frame = cvCreateImage(cvSize(widVideo,htVideo),IPL_DEPTH_8U,1);

	sem_wait( &sem_filled ) ;	// block here wait for the reader to be ready 

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

		current_frame = cvQueryFrame( videoIn ) ;	
		frame_cnt++ ;

#ifdef CUDA_OBS 
		cvShowImage( "Display Camera CCC", current_frame ) ;

		cvWaitKey( 10 ) ;
#endif 

		memcpy ( fp->vbp + sizeVideo * fp->cnt, current_frame->imageData,
			sizeVideo ) ;

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

		fp->cnt++ ;

		// sleep( 1 ) ;

		if ( fp->cnt == frame_per_io )
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
cs_ipcam_get()
{
	struct frame_list *fp ;

	fp = frame_get( &filled_frame_head ) ;

#ifdef CUDA_OBS 
	printf("cs_ipcam_get *** fp %p vbp %p gbp %p \n", fp, fp->vbp, fp->gbp ) ;
#endif 

	return ( fp ) ;
}

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

void
cs_ipcam_put( struct frame_list *fp )
{
#ifdef CUDA_OBS 
	int *dp, i, j, h, v, t, va, oh, ov, ot, ova ;
#endif 

	printf("cs_ipcam_put: fp %p outp %p ----------------------------------------\n",
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
			va = *dp++ ;

			ot = *dp++ ;
			ov = *dp++ ;
			oh = *dp++ ;
			ova = *dp++ ;
			

			printf("BLK %d %d : t %d v %d h %d va %d, ot %d ov %d oh %d ova %d\n",
				j, i, t, v, h, va, ot, ov, oh, ova ) ;
		}
	}

	printf("END ================================================================\n") ;
#endif 
	
	// the pic based on what is in outp first ... then return the whole frame_list
	frame_put( &display_frame_head, fp ) ;
}

void
cs_ipcam_start()
{
	sem_post( &sem_filled ) ;	// ok ... free the grabber ...
}

void
cs_ipcam_record ( int *fp, int *tp ) 
{
	memcpy ( tp, fp, blk_x * blk_y * OUTP_ENTRY_SIZE * sizeof ( int )) ;
}

#ifdef CUDA_OBS 
void * // just grab and free ... for testing purpose
frame_server( void *p)
{
	struct frame_list *fp ;
	IplImage *current_frame = 0 ;
	int new_frame = 1 ;
	int frame_cnt = 0, i ;

	for ( i = 0 ; i < 1000000 ; i++ )
	{
		if ( new_frame )
		{
			printf("getting P\n") ;

			new_frame = 0 ;
			fp = frame_get ( &filled_frame_head ) ;
			printf("P: fp %p vbp %p gbp %p \n", fp, fp->vbp, fp->gbp ) ;
			fp->cnt = 0 ;
		}

		fp->cnt++ ;

		frame_cnt++ ;
		printf("=== fr %d\n", frame_cnt ) ;

		if ( fp->cnt == frame_per_io )
		{
			new_frame++ ;
			frame_put ( &free_frame_head, fp ) ;
			printf("=") ;
		}
	}
}
#endif 

// take off display_frame_head, put to free_frame for testing purpose

static void
do_frame_displayer( void *p)
{
	struct frame_list *fp ;
	IplImage *current_frame = 0 ;
	int xpos, ypos, frame_cnt = 0 ;
	int *dp, i, j, k, h, v;
#ifdef CUDA_OBS
	int t, va ;
#endif

	// cvNamedWindow("Display Camera PPP", CV_WINDOW_AUTOSIZE ) ;

	if ( displayGray )
		current_frame = cvCreateImage(cvSize(widVideo,htVideo),IPL_DEPTH_8U,1);
	else
		current_frame = cvCreateImage(cvSize(widVideo,htVideo),IPL_DEPTH_8U,3);

	while ( 1 )
	{
		fp = frame_get ( &display_frame_head ) ;
#ifdef CUDA_OBS 
		printf("frame_displayer *** fp %p vbp %p gbp %p \n", fp, fp->vbp, fp->gbp ) ;
#endif 

		for ( k = 0 ; k < frame_per_io ; k++ )
		{
			if ( displayGray )
				current_frame->imageData = fp->gbp + sizeGray * k ;
			else
				current_frame->imageData = fp->vbp + sizeVideo * k ;

#ifdef CUDA_OBS 
			printf("frame_displayer === fp %p vbp %p gbp %p k %d\n",
				fp, fp->vbp, fp->gbp, k ) ;
#endif 

			dp = fp->outp ;
			ypos = 0 ;

			for ( i = 0 ; i < blk_y ; i++ )
			{
				xpos = 0 ;
				for ( j = 0 ; j < blk_x ; j++ )
				{
#ifdef CUDA_OBS
					t = *dp++ ;
#else
					dp++;
#endif
					v = *dp++ ;
					h = *dp++ ;
#ifdef CUDA_OBS
					va = *dp++ ;
#else
					dp++;
#endif

					dp += 4 ;	// should have a better name ...

#ifdef CUDA_OBS 
					if (( va < 0 ) && 
						(( abs ( h - md_x ) > disp_th_x) || ( abs( v - md_y ) > disp_th_y ))) 
#endif 
					if (( abs ( h - md_x ) > disp_th_x) || ( abs( v - md_y ) > disp_th_y ))
					{
#ifdef CUDA_OBS 
						printf("frame_displayer: i %d j %d t %d v %d h %d va %d x/y %d %d\n",
							i, j, t, v, h, va, xpos, ypos ) ;
#endif 

						drawArrow( current_frame, cvPoint(xpos,ypos),
							cvPoint( xpos + ( h - md_x ) * 6 , ypos + ( v - md_y ) * 6 ),
							CV_RGB(0,0,0)) ;
					}

					xpos += display_step_x ;
				}
				ypos += display_step_y ;
			}

			cvShowImage( "Display Camera PPP", current_frame ) ;

			cvWaitKey( ms_wait ) ;

			frame_cnt++ ;

		}

		frame_put ( &free_frame_head, fp ) ;
#ifdef CUDA_OBS 
		printf("frame_displayer: cnt %d\n", frame_cnt ) ;
#endif 
	}
}

static void *
frame_displayer( void *p)
{
  do_frame_displayer(p);
  return NULL;
}
