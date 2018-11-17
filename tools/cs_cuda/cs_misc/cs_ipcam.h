#ifndef __CS_IPCAM_H__
#define __CS_IPCAM_H__

struct frame_list {
	char 	*vbp ;
	char 	*gbp ;

	int 	*outp ; // for the result of analysis

	// used by owner of this struct
	int cnt ;

	struct frame_list *np ;
} ;


#define OUTP_ENTRY_SIZE		8
/*
t/v/h/va : for best score
t/v/h/va : for no motion ... cf. Razi's doc
*/ 

int cs_ipcam_init( int block_z, int width, int height, char *ipcam, int blk_x, int blk_y,
	int md_x, int md_y, int th_x, int th_y ) ;
struct frame_list *cs_ipcam_get() ;
void cs_ipcam_put( struct frame_list *fp ) ;
void cs_ipcam_start() ;
void cs_ipcam_record ( int *fp, int *tp ) ;

#endif 
