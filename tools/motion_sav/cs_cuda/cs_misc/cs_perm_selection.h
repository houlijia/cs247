#ifndef __CS_PERM_SELECTION_H__
#define __CS_PERM_SELECTION_H__

void h_do_perm_selection_L ( int *d_perm_tbl, int tbl_size,
	int *d_perm_tbl_cube, int cube_size, int random, int sink ) ;

void h_do_perm_selection_R ( int *d_perm_tbl, int tbl_size,
	int random ) ;

void h_do_get_perm_matrix( int *dp, int ox, int oy, int oz, int cx,
	int cy, int cz, int *sink ) ;

int h_do_find_perm_size( int ox, int oy, int oz, int *cx,
	int *cy, int *cz, int max_z, int nmea, int min_x, int min_y ) ;

#endif 
