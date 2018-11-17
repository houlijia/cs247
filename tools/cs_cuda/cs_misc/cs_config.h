#ifndef __CS_CONFIG_H__
#define __CS_CONFIG_H__

#define PATH_LENG	256

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
	char ipcam_string[PATH_LENG] ;

	// -i : input data file
	char finname[PATH_LENG] ;

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

	// do reconstruction ;

	int do_reconstruction ;

	// global -- not configurable ... derived
	
	int do_analysis ;
	int do_block	;
	int do_comp_ratio ;
	int do_display ;
	int do_interpolate ;

} ;

void cs_config_p ( struct cs_config * ) ;
void cs_config_init( struct cs_config *csp ) ;
int cs_config( char *jsonfile, struct cs_config *csp ) ;

#endif 
