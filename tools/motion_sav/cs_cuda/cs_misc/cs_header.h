#ifndef __CS_HEADER_H__ 
#define __CS_HEADER_H__ 

#include "cs_analysis.h"

struct cs_xy {
	int x ;
	int y ;
} ;

struct cs_xyz {
	int x ;
	int y ;
	int z ;
} ;

struct cs_header {
	char 	coding ;
	char	coding_opt ;
	char	matrix ;
	char	weight ;

	// frame x y
	struct cs_xy frame ;

	// block x y
	struct cs_xyz	block ;

	// cube for selection
	struct cs_xyz	select[ CUBE_INFO_CNT ] ;

	// overlap
	struct cs_xyz 	overlap ;

	// expansion
	struct cs_xyz	expand ;

	// append in block
	struct cs_xy	append ;

	// edge
	struct cs_xy	edge ;

	// motion detection
	struct cs_xyz	md ;

	int	pad[ 1 ] ;
} ;


// cs_header.coding

#define CS_CD_YUV420P	1

// cs_header.coding_opt

#define Y_COMP_ONLY		0x1 // for CS_CD_YUV420P
#define DOUBLE_PERM		0x2 // for double_permutation ... Razi
#define ML_PERM			0x4 // for maximum length permutation ... Razi
#define CS_WEIGHT		0x8 // for all
#define CS_OVERLAP		0x10 // for all
#define CS_CO_BIGENDIAN		0x80 // for all

// cs_header.matrix

#define WALSH_HADAMARD_MATRIX		1

int 
cs_put_header( int fd, char coding, char opt, char m,
	int x, int y,
	int xb, int yb, int zb,
	int xc0, int yc0, int zc0,
	int xc1, int yc1, int zc1,
	int xc2, int yc2, int zc2,
	int xo, int yo, int zo,
	int xe, int ye, int ze,
	int xa, int ya,
	int edge_x, int edge_y,
	int md_x, int md_y, int md_z,
	int weight ) ;

int cs_get_header( int fd, struct cs_header *cshp ) ;

// block header

struct cs_block_header {
	int random_r ;
	int random_l ;

	int pad[2] ;
} ;

int cs_put_block_header( int fd, int randoml, int randomr )  ;
int cs_get_block_header( int fd, struct cs_block_header *cshp ) ;

#endif 
