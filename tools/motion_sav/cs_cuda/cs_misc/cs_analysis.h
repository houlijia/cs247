#ifndef __CS_ANALYSIS_H__
#define __CS_ANALYSIS_H__

struct cube {
	int *dp ;	// on device, the selection cube info.
				// has index to the location of the same element in the block
				// ck the allocate_d_mem.cudaMalloc( ... )
	int x ;
	int y ;
	int z ;
	int size ;
	int sink ;

	int md_v3_cnt ; // v4: how many hvt block for this block, after edge-detection
	int md_v3_hv_cnt ; // v4: num of T-elements, of hv-size,  in this record

	int md_v4_loopcnt ; // number of record in this block, include the 1 for orig record
	// loopcnt = md_v3_cnt * ( md_x * 2 + 1 ) * ( md_y * 2 + 1 ) + 1
	int md_v4_record_length ; // include the HVT 

	float interval ; // for quantization

	int *cube_perm ;	// on device, final L-permutation table for this cube
} ;

// 0:inner, 1:side, 2:corner 

#define CUBE_INFO_CNT 3
#define CUBE_INFO_INNER			0
#define CUBE_INFO_SIDE			1
#define CUBE_INFO_CORNER		2

#define CUBE_INFO_SHIFT			30
#define CUBE_INFO_MSK			0x3
#define CUBE_INFO_T_MSK			0xff	
#define CUBE_INFO_GET(x)		(( x >> CUBE_INFO_SHIFT ) & CUBE_INFO_MSK )		

#define CUBE_INFO_SET(x)		(( x & CUBE_INFO_MSK ) << CUBE_INFO_SHIFT )

struct per_blk_data {

	float 	*d_meanp ;
	float 	*d_sdp ;
	float 	*d_dcp ;
	float 	*d_interval ;	// needed?

	float	*d_ampl ;	
	float	*d_offset ;

	int		*d_max_bin ;
	int		*d_num_bin ;
} ;

#define NUM_F_IN_PBD	6
#define NUM_I_IN_PBD	2


#endif 

