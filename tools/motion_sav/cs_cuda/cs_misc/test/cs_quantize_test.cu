
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
#include "cs_matrix.h"
#include "cs_decode_parser.h"
#include "cs_quantize.h"

#define CUDA_DBG
#define BUF_SIZE_INT		10
#define BUF_SIZE_INT1		8	

int *dp1, *dp2, *dp3 ;
int *dp11, *dp22, *dp33 ;

int buf1[ BUF_SIZE_INT ] ; 
float dbuf[ BUF_SIZE_INT ] ;

float fres, *fp1 ;
float hd1[] = { 0.2, 0.4, 0.6 } ;
float hd2[] = { 1.2, 1.4, 1.6 } ;

struct CS_EncParams CS_EncParams ; 	// 1
struct RawVidInfo RawVidInfo ;	// 2
struct VidRegion VidRegion ;	// 3
struct SensingMatrixWH SensingMatrixWH ;	// 4
struct UniformQuantizer UniformQuantizer ;	// 5
struct QuantMeasurementsBasic QuantMeasurementsBasic ; // 6
int msr_idx[10000 ] ;

int
main( int ac, char *av[] )
{
	int *d_int, i, type, k ;
	int first = 1 ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	QuantMeasurementsBasic.h_msr_idxp = msr_idx ;

	cudaMalloc ( &d_int, 10000 * sizeof ( int )) ;

	if ( ac != 2 )
	{
		printf("usage : %s csvid_file\n", av[0] ) ;
		exit( 2 ) ;
	}

	if (! cs_decode_parser_init( av[1], 10000 ))
	{
		printf("%s : failed init\n", __func__ ) ;
		exit( 3 ) ;
	}

	k = 10000 ;
	while ( k-- )
	{
		i = get_next_type ( &type ) ;

		if ( i == 0 )
		{
			printf("%s : failed type \n") ;
			exit( 3 ) ;
		} else if ( i < 0 )
		{
			printf("%s : eof\n", __func__ ) ;
			exit( 0 ) ;
		}
	

		printf("TYPE %d --------------------------------------------------\n", type ) ;

		switch ( type ) {
		case 1 :
			if ( !get_next_element ( type, ( void *)&CS_EncParams ))
				exit( 3 ) ;

			p_element( type, "from 1", ( void *)&CS_EncParams ) ;
			break ;

		case 2 :
			if ( !get_next_element ( type, ( void *)&RawVidInfo ))
				exit ( 2 ) ;
			p_element( type, "from 2", ( void *)&RawVidInfo ) ;
			break ;

		case 3 :
			if ( !get_next_element ( type, ( void *)&VidRegion ))
				exit ( 2 ) ;
			p_element( type, "from 3", ( void *)&VidRegion ) ;
			break ;

		case 4 :
			if ( !get_next_element ( type, ( void *)&SensingMatrixWH ))
				exit ( 2 ) ;

			p_element( type, "from 4", ( void *)&SensingMatrixWH ) ;

			break ;

		case 5 :
			if ( !get_next_element ( type, ( void *)&UniformQuantizer ))
				exit ( 2 ) ;
			p_element( type, "from 5", ( void *)&UniformQuantizer ) ;
			break ;

		case 6 :
			if ( !get_next_element ( type, ( void *)&QuantMeasurementsBasic ))
				exit ( 2 ) ;
			p_element( type, "from 6", ( void *)&QuantMeasurementsBasic ) ;

			put_d_data_i ( d_int, msr_idx, sizeof ( int ) * QuantMeasurementsBasic.lenb ) ;

			h_do_unquan_adj_index ( d_int, QuantMeasurementsBasic.lenb, QuantMeasurementsBasic.noclip,
				QuantMeasurementsBasic.nbin/2 -1,
				QuantMeasurementsBasic.nbin ) ;

			if ( first )
			{
				first = 0 ;
				cs_decode_parser_reinit ( 12000 ) ;
			}
			break ;

		default :
			printf("%s :: wrong default type %d \n", type ) ;
			exit( 4 ) ;
		}
	}
}
