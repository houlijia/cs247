#ifndef __CS_MOTION_DETECT_H__
#define __CS_MOTION_DETECT_H__

int h_do_motion_idx ( int *dp, int total_size, int record_length,
	int h_loop, int v_loop, int t_loop, int *orig ) ;

int h_do_motion_detection ( int *fromp, int *top,
	int tbl_size, int record_size,
	int blk_x, int blk_xy,	
	int cube_x, int cube_xy ) ;

void h_do_l1_norm_step1( int *dp, int total, int record_size, int orig ) ;
int h_do_l1_norm_step2( int *dp, int total, int record_size ) ;
int h_do_l1_norm_step3( int *dp, int total, int record_size, int orig ) ;
int h_do_l1_norm_step4( int *dp, int total, int record_size, int orig,
	int * ) ;

#endif 
