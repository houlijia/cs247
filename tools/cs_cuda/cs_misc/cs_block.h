#ifndef __CS_BLOCK_H__
#define __CS_BLOCK_H__

#define NO_WEIGHT 	0		
#define WEIGHT_LINEAR 	0x3	// if width is 2b, the weight goest from
		// 1, 2, ..., b, b, b-1, b-2, ..., 2, 1,

void h_make_block ( int *d_input, int *d_output,
	int xdim, int ydim,
	int frame_size,
	int xbdim, int ybdim, int zbdim, int blk_dst_size,
	int do_perm,
	int x_overlap, int y_overlap,
	int x_blknum, int y_blknum, int app_x, int app_y, int weight_scheme,
	int shift
#ifdef CUDA_OBS 
	, int *cudadbgp
#endif 
	) ;

#endif 
