#include <iostream>
using namespace std;

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_dct.h"

#define CUDA_DBG

float *dp1 = NULL, *dp2 = NULL ;

#define NUM_OF_HVT_INDEX 3

#define BUF_SIZE	( 1024 * 1024 )
#define BUF_SIZE_INT	( BUF_SIZE * sizeof (int))

#define X_SIZE		3
#define Y_SIZE		4	
#define T_SIZE		3
#define MTRX_SIZE	( X_SIZE * Y_SIZE * T_SIZE )

float randi_100_2_6[] = {		// 2x3x2
	82.0, 13.0, 64.0, 
	28.0, 96.0, 16.0, 
	91.0, 92.0, 10.0,
	55.0, 97.0, 98.0 } ;

/* dct of randi_100_2_6 

122.3295   74.2462   52.3259   
58.6899  136.4716   80.6102

-6.3640  -55.8614   38.1838  
-19.0919   -0.7071  -57.9828

*/

float randi_100_3_12[] {	// 3 x 4 x 3
	69.0,     9.0,    16.0,   100.0,    11.0,    78.0,     9.0,    81.0,    19.0,    14.0,    55.0,    63.0,  
	75.0,    23.0,    83.0,     8.0,    97.0,    82.0,    40.0,    44.0,    27.0,    87.0,    15.0,    36.0,  
	46.0,    92.0,    54.0,    45.0,     1.0,    87.0,    26.0,    92.0,    15.0,    58.0,    86.0,    52.0 
} ;	


/*
   dct of randi_100_3_12[] 
109.6966   71.5914   88.3346   88.3346   62.9312  142.6055   43.3013  125.2850   35.2184   91.7987
16.2635  -58.6899  -26.8701   38.8909    7.0711   -6.3640  -12.0208   -7.7782    2.8284  -31.1127
-14.2887   22.4537  -39.1918   52.6640  -74.3012    0.4082  -18.3712   34.7011   -8.1650  -41.6413

90.0666   87.1799
-21.9203    7.7782
45.3156   17.5547
*/



float buf1[MTRX_SIZE] ; 

struct cs_xyz	hcube[ CUBE_INFO_CNT ], *dcubep ;
struct cube cubecube[ CUBE_INFO_CNT ] ;

int
main( int ac, char *av[] )
{
	int k ;
	float *fp, *dp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;
	h_do_dct_init() ;

	if (( k = cudaMalloc( &dp1, BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	if (( k = cudaMalloc( &dp2, BUF_SIZE_INT )) != cudaSuccess )
	{
		printf("%s: cube alloc failed %d \n", __func__, k ) ;
		exit ( 0 ) ;
	}

	if ( !h_do_dct_init())
	{
		printf("dct init failed") ;
		exit ( 0 ) ;
	}

	dp = buf1 ;

#ifdef CUDA_DBG 
	// setup data for 2x3x2 case ...

	// fp = randi_100_2_6 ;
	fp = randi_100_3_12 ;
	k = MTRX_SIZE ;
	for ( k = 0 ; k < MTRX_SIZE ; k++ )
		*dp++ = *fp++ ;
#endif 

	dbg_put_d_data (( char *)dp1, ( char *)buf1, sizeof( float ) * MTRX_SIZE ) ; 

	// dct

	dbg_p_d_data_i_cube ( "before dct", dp1, X_SIZE, Y_SIZE, T_SIZE ) ;

	if ( !h_do_dct( dp1, dp2, X_SIZE, Y_SIZE, T_SIZE, 0 ))
	{
		printf("%s: h_do_dct failed \n", av[0] ) ;
		exit ( 3 ) ;
	}
  
	dbg_p_d_data_i_cube ( "after dct", dp2, X_SIZE, Y_SIZE, T_SIZE ) ;

	// idct 

	dbg_p_d_data_i_cube ( "before idct", dp1, X_SIZE, Y_SIZE, T_SIZE ) ;

	if ( !h_do_dct( dp1, dp2, X_SIZE, Y_SIZE, T_SIZE, 1 ))	// inverse
	{
		printf("%s: h_do_dct failed \n", av[0] ) ;
		exit ( 3 ) ;
	}
  
	dbg_p_d_data_i_cube ( "after idct", dp2, X_SIZE, Y_SIZE, T_SIZE ) ;

}
