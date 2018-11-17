#ifndef __CS_MOTION_DETECT_V3_H__
#define __CS_MOTION_DETECT_V3_H__

// FIX ... this logic depends on the size of float and int are the same
int h_do_motion_idx_v3 ( int *dp, int total_size,
	int *orig_idx, struct cube *cubep,
	int md_x, int md_y, int md_z, int *record_sizep, int cube_idx ) ;

template<typename T> int
h_do_motion_detection_step0_v3 ( T *fromp, T *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int edge_x, int edge_xy,	// orig edged block x/y size
	int md_x, int md_xy,	// new md block x/y size
	int md_v3_cnt ) ;	// max number of in_x*in_y in a record

template<typename T> int
h_do_l1_norm_step1_v3( T *dp, int total, int record_size, int orig ) ;

template<typename T> int
h_do_l1_norm_step2_v3( T *dp, int record_size, int row ) ;

template<typename T> int
h_do_l1_norm_step3_v3( T *dp, int record_size, int orig, int row ) ;

template<typename T> int
h_do_l1_norm_step4_v3( T *dp, int record_size, int orig,
	int row_size, int *resp, int no_motion_idx, int omp_idx ) ;

#define NUM_OF_HVT_INDEX	3
#define L1_NORM_STEP4_RETURN_ENTRY_SIZE		4
	// 0: T(with flag), 1:V, 2:H, 3:value	1-L1-norm, normalized, 

#endif 

