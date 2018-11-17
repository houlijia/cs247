#ifndef __CS_EDGE_DETECT_V2_H__
#define __CS_EDGE_DETECT_V2_H__

int h_do_edge_detection_v2 ( int *fromp, int *top, int tbl_size,
	struct cs_xyz *d_xyzp, int edge_x, int edge_y, int blk_in_x,
	int blk_in_y, struct cube *cubep ) ;

#endif 
