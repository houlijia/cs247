#ifndef __CS_VIDEO_IO_H__
#define __CS_VIDEO_IO_H__

struct frame_list {
	char 	*vbp ;
	char 	*gbp ;

	int 	*outp ; // for the result of analysis

	// used by owner of this struct
	// for VIDEO_SRC_FILE, if cnt == 0 after cs_vio_get(), then the EOF is reached.
	int cnt ; 

	struct frame_list *np ;
} ;


#define OUTP_ENTRY_SIZE		8
/*
t/v/h/va : for best score
t/v/h/va : for no motion ... cf. Razi's doc
*/ 

int cs_vio_init( int frame_in_block, int width, int height, char *axis, int blkx, int blky,
	int md_xx, int md_yy, int display_threshold_x, int display_threshold_y, int video_source,
	int frame_rate, int do_display, int ignore_edge, int ignore_arrow, float display_threshold ) ;

struct frame_list * cs_vio_get() ;
void cs_vio_put( struct frame_list *fp ) ;
void cs_vio_start() ;
void cs_vio_record ( int *fp, int *tp ) ;

#endif 
