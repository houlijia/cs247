#ifndef __CS_COPY_BOX_H__
#define __CS_COPY_BOX_H__

int h_do_copy_box ( int *fromp, int *top, int tbl_size, int cube_x,
	int cube_y, int edge_x, int edge_y ) ;

/*
   this routine copy the cubes in the all the blocks into the vector
   pointed by top.  from_size is the block size, to_size is the cube size
	total_size is the size of the copy ... in element
	copy the first from_size elements from fromp to top for every block
*/
template<typename T>
int
h_do_copy_vec ( T *fromp, T *top, int total_size, int from_size,
		int to_size );


template<typename T> int
h_do_copy_box_v2 ( T *fromp, T *top, int tbl_size, 
	int edge_x, int edge_y, int blk_in_x, int blk_in_y, struct cube *d_cp, 
	struct cube *cp ) ;


#endif 
