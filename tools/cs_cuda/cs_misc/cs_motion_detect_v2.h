#ifndef __CS_MOTION_DETECT_V2_H__
#define __CS_MOTION_DETECT_V2_H__

int h_do_motion_idx_v2 ( int *dp, int total_size,
	int *orig_idx,
	int blk_in_x, int blk_in_y, struct cube *cubep,
	int md_x, int md_y, int md_z, int *record_size ) ;

int h_do_motion_detection_step0_v2 ( int *fromp, int *top,
	int tbl_size,
	int record_size,
	int md_x, int md_y, int md_z,
	struct cs_xyz *d_xyzp,
	int hvt_size, int from_block_size ) ;

int h_do_l1_norm_step1_v2( int *dp, int total, int record_size,
	int orig, int hvt_size) ;

int h_do_l1_norm_step2_v2( int *dp, int total, int record_size,
	struct cube *hcubep, struct cs_xyz *d_xyzp, int *d_resp ) ;

int h_do_l1_norm_step3_v2( int *dp, int total, int record_size,
	int orig, int hvt_size ) ;

int h_do_l1_norm_step4_v2( int *dp, int total, int record_size,
	int orig, int hvt_size, int *v, int no_motion_idx ) ;

#define NUM_OF_HVT_INDEX	3
#define L1_NORM_STEP4_RETURN_ENTRY_SIZE		4
	// 0: T(with flag), 1:V, 2:H, 3:value	1-L1-norm, normalized, 

#endif 
