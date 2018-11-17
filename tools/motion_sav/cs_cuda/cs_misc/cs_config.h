#ifndef __CS_CONFIG_H__
#define __CS_CONFIG_H__

#define PATH_LENG	256

#define VIDEO_SRC_IPCAM		1
#define VIDEO_SRC_WEBCAM	2
#define VIDEO_SRC_FILE		3


struct cs_config {
	// -a : adjustment of the block along top and right side of the block with 0
	int adj_x ;
	int adj_y ;

	// -f : debug flag ;
	unsigned int dbg_flag ;

	// -p : do permutation 
	int do_permutation ;
	char permdir[PATH_LENG] ;

	// -I : ip cam string
	char video_src[PATH_LENG] ;
	int video_source ; // 1: ipcam 2:webcam 3:file

	// -i : input data file
	// char finname[PATH_LENG] ; when video_source is 3, video_src is the finname

	// 1: display the output image
	int	do_display ;

	// ignore the edge ... do not display the arrows of motion dection around the edges of the frame
	int ignore_edge ;

	// frame_per_second
	int	fps ;

	// -z : comp_ratio
	int	comp_ratio ;

	// -m : motion detection
	int md_x ;
	int md_y ;
	int md_z ;

	// -c : cube
	int cubex ;
	int cubey ;
	int cubez ;

	int do_cube ; // derived

	// -e : frame expansion
	int xadd ;
	int yadd ;
	int zadd ;

	// -g : edge
	int edge_x ;
	int edge_y ;

	// -T : display threshold 
	int disp_th_x ;
	int disp_th_y ;

	// -d : frame size
	int frame_x ;
	int frame_y ;

	// -n : do not seek
	int do_not_seek ;

	// -F : md output file
	char md_outputfile[PATH_LENG] ;

	// -o : output file name
	char foutname[PATH_LENG] ;

	// -q
	int do_one ;

	// -s
	int do_swap ;

	// -y
	int y_only ;

	// -O : overlap
	int overlap_x ;
	int overlap_y ;
	int overlap_z ;

	// -w : weight scheme
	int weight_scheme ;

	// -b : block size
	int x_block ;
	int y_block ;
	int z_block ;

	// reconstruction 
	int do_reconstruction ;

	// display_threshold ;
	float display_threshold ;

	// capture the video, if not 0, it indicates the frame cnt we are going
	// to capture.  it will only capture, no other opeartion is done. 
	int capture ;

	int do_analysis ;

	// quantization
	int interval_factor[3] ;	// INNER, SIDE, CORNER

	// global -- not configurable ... derived
	
	int do_block	;
	int do_comp_ratio ;
	int do_interpolate ;

} ;

void cs_config_p ( struct cs_config * ) ;
void cs_config_init( struct cs_config *csp ) ;
int cs_config( char *jsonfile, struct cs_config *csp ) ;

#endif 
