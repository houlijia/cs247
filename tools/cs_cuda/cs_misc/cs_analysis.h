#ifndef __CS_ANALYSIS_H__
#define __CS_ANALYSIS_H__

struct cube {
	int *dp ;	// on device, the selection cube info
	int x ;
	int y ;
	int z ;
	int size ;
	int sink ;

	int *cube_perm ;	// on device
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

#endif 

