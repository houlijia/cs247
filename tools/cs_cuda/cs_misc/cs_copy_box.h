#ifndef __CS_COPY_BOX_H__
#define __CS_COPY_BOX_H__

int h_do_copy_box ( int *fromp, int *top, int tbl_size, int cube_x,
	int cube_y, int edge_x, int edge_y ) ;

int h_do_copy_vec ( int *fromp, int *top, int total_size, int from_size,
	int to_size ) ;

int h_do_copy_box_v2 ( int *fromp, int *top, int tbl_size, 
	int edge_x, int edge_y, int blk_in_x, int blk_in_y, struct cs_xyz *cp,
	struct cube * ) ;

#endif 
