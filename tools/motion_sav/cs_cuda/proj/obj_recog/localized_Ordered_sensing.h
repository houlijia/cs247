#ifndef __LOCALIZED_ORDERED_SENSING_H__
#define __LOCALIZED_ORDERED_SENSING_H__

int * lo_sensing( int *ip, int orig_w, int orig_h, int w, int h, int w_offset,
	int h_offset )  ;

struct obj_param {
	int	w ;	// size of width ... the same as the height
	int size ; // should be w * w * 6
	int image_size ; // should be w * w * 16 
} ;

struct obj_rec_param {
	int tag ;	// see below
	int size ;	// the image returned by the obj_recog
	int status ;	// see below
} ;

// obj_rec_param.tag
#define TAG_2		0x8badf00d

// obj_rec_param.status
#define OBJ_RECOG_GOOD		0
#define OBJ_RECOG_ALARM		1
#define OBJ_RECOG_ERR		2

#define MARKER		0x5ee5e55e

#endif 
