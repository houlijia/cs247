#ifndef __CS_MOTION_DETECT_V4_H__
#define __CS_MOTION_DETECT_V4_H__

// FIX ... this logic depends on the size of float and int are the same
int
h_do_motion_idx_v4 ( int *dp, int total_size,
	int blk_in_x, int blk_in_y, struct cube *cubep,
	int md_x, int md_y, int md_z, struct cube *d_cubep ) ;

int
set_up_cube_log( struct cube *tcubep, int hvt_adj );

int
set_up_cube_log_cont ( struct cube *tcubep );

template<typename T> int
h_do_motion_detection_step0_v4 ( T *fromp, T *top,
	int tbl_size,		// overall input size ... excludes the 3 indexes
	int md_x, int md_y, int md_z,
	struct cube *d_cubep,	// cube in device	// will have the size of the 
	int from_block_size,
	int to_blk_entries_size, // exclude tvh ...
	int to_blk_size,	// include the tvh
   	int nblk_in_x, int nblk_in_y ) ; // new

template<typename T> int
h_do_l1_norm_step1_v4( T *dp, int total, int entries_per_block, int blk_size,
	struct cube *d_cubep, int blk_in_x, int blk_in_y,
    int max_record ) ;

template<typename T> int
h_do_l1_norm_step2_v4( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size, int hvt_adj ) ;

template<typename T> int
h_do_l1_norm_step2_v4_block( T *dp, struct cube *h_cubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template<typename T> int
h_do_l1_norm_step2_v4_block_v1( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template<typename T> int
h_do_l1_norm_step2_v4_block_v2( T *dp, struct cube *hcubep, struct cube *d_cubep, 
	int blk_in_x, int blk_in_y, int blk_size ) ;

template<typename T>
int
h_do_l1_norm_step3_v4( T *dp, int total, int entries_per_block, 
	struct cube *d_cubep, int to_blk_size, int blk_in_x, int blk_in_y ) ;

template<typename T>
int
h_do_l1_norm_step4_v4( T *dp, int total,
	int *resp, int no_motion_idx,
	struct cube *d_cubep,
	int to_blk_size,
	int blk_in_x, int blk_in_y,
	struct cube *h_cubep) ;


// pflag
#define P_TVH_IDX			0x1
#define P_BLK				0x2 // p1 has the block number
								// p2 << p3 ... record
#define P_ROW				0x4	// p1 has the block number
								// p2 << p3 ... record ... 


int h_dbg_md_v4 ( const char *s, float *dp, struct cube *cubep, int blk_size,
	int blk_in_x, int blk_in_y, int pflag, int p1, int p2, int p3 ) ;

#define NUM_OF_HVT_INDEX	3
#define L1_NORM_STEP4_RETURN_ENTRY_SIZE		4
	// 0: T(with flag), 1:V, 2:H, 3:value	1-L1-norm, normalized, 

__device__ int get_blk_type_idx ( int blk_idx, int blk_in_x, int blk_in_y ) ;


__device__ int d_max_log2( int i );

int find_thread_blk ( int threads );

#endif 
