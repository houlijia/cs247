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
#include "cs_whm_encode.h"
#include "cs_header.h"

#define CUDA_DBG 

#ifdef CUDA_DBG 
#define YUV_BUF_SIZE 102400 
char yuv_dbg[ YUV_BUF_SIZE ] ;
#endif 

void fix_it ( int fin, int fout ) ;
int do_measurement ( int fin, int fout, int x, int y ) ;
int make_one_component( int fin, int fout,
	char *din_a, int *dout_a,
	char *hin_a, char *hout_a,
	int ysize, int dout_size, int din_size ) ;

int y_only = 1 ;

main( int ac, char *av[] )
{
	int x, y, fin, fout ;

	if ( ac < 5 )
	{
		printf("Usage: %s yuv-file-in X Y yuv-file-out [0=A,[1]=Y]\n", av[0]) ;
		exit( 1 ) ;
	}

	if ( ac >= 6 )
		y_only = atoi ( av[5] ) ;

	x = atoi ( av[2] ) ;
	y = atoi ( av[3] ) ;

	fin = open( av[1], O_RDONLY ) ;

	if ( fin == -1 )
	{
		printf("file %s does not exist\n", av[1]) ;
		exit( 1 ) ;
	}

	fout = open( av[4], O_CREAT | O_TRUNC | O_WRONLY, S_IRWXU ) ;

	if ( fout == -1 )
	{
		printf("file %s open failed %d\n", av[1], errno ) ;
		exit( 1 ) ;
	}

	if ( !cs_put_header ( fout, CS_CD_YUV420P, ( y_only )? Y_COMP_ONLY : 0,
		WALSH_HADAMARD_MATRIX, x, y, x, y, 1 ))
	{
		printf("can't write header\n") ;
		exit( 1 ) ;
	} 

	if ( !do_measurement( fin, fout, x, y ))
	{
		printf("do_measurement: failed\n") ;
	    exit( 1 ) ;
	}	

	close ( fin ) ;
	close ( fout ) ;
}

int
do_measurement ( int fin, int fout, int x, int y )
{
	int y2size, u2size, yloop, uloop, ysize, usize, i, din_size, dout_size,
		total, frame_cnt ;
	char *buf_in, *buf_out ;
	char *d_in_p ;
	int *d_out_p ;
	double d, ad ;
	int cnt ;

	din_size = max_log2( x * y ) ; 
	dout_size = din_size * sizeof ( int ) ;

	yloop = ( int )log2(( double ) din_size ) ;
	uloop = yloop - 2 ; // uloop and vloop are the same. u/v component individually 

	ysize = x * y ;
	usize = ysize >> 2 ;

	y2size = ( int ) pow ( 2.0, yloop ) ;
	u2size = ( int ) pow ( 2.0, uloop ) ;

	buf_in = ( char * ) malloc ( din_size ) ;
	if ( buf_in == NULL )
	{
		printf("do_measurement: malloc failed \n") ;
		return ( 0 ) ;
	}

	buf_out = ( char * ) malloc ( dout_size ) ;
	if ( buf_out == NULL )
	{
		printf("do_measurement: 2 malloc failed \n") ;
		return ( 0 ) ;
	}

	if ( cudaMalloc( &d_in_p, din_size ) != cudaSuccess )
	{
		printf("do_measurement: cudaMalloc failed \n") ;
		return ( 0 ) ;
	}

	if ( cudaMalloc( &d_out_p, dout_size ) != cudaSuccess )
	{
		printf("do_measurement: 2 cudaMalloc failed \n") ;
		return ( 0 ) ;
	}

	fprintf(stderr, "yonly %d i %d o %d usize %d ysize %d yloop %d uloop %d\n",
		y_only, din_size, dout_size, usize, ysize, yloop, uloop ) ;

	dbg_init ( 4 * 1024 * 1024 ) ;

	// clear d_in_mem 

	// dbg_p_d_data_c ( "== before clear", d_in_p, din_size ) ;

	clear_device_mem_c( d_in_p, din_size ) ;

	// dbg_p_d_data_c ( "after clear in", d_in_p, din_size ) ;

	omp_timer_init( 1 ) ;

	frame_cnt = 0 ;
	total = 0 ;
	while ( 1 )
	{
		// do y

		i = make_one_component( fin, fout, d_in_p, d_out_p,
			buf_in, buf_out, ysize, y2size << 2, y2size ) ;

		if ( !i )
		{
			printf("do_measurement:failed total %d\n", total ) ;
			return ( 0 ) ;
		}

		if ( i == 2 )
		{
			printf("do_measurement: total %d frame_cnt %d \n",
				total, frame_cnt ) ;
			omp_timer_get ( 0, &d, &cnt, &ad ) ;
			printf("overall %f cnt %d average %f \n", d, cnt, ad ) ;
			return ( 1 ) ;
		}

		total += ysize ;

		if ( y_only )
		{
			if (( i = lseek ( fin, usize << 1, SEEK_CUR )) < 0 )
			{
				printf("do_measurement: failed lseek %d total %d \n",
					errno, total ) ;
				return ( 0 ) ;
			}

			total += ( usize << 1 ) ;
		} else
		{
			clear_device_mem_c( d_in_p, u2size ) ;

			i = 2 ;
			while ( i-- )
			{
				if ( !make_one_component( fin, fout, d_in_p, d_out_p,
					buf_in, buf_out, usize, u2size << 2, u2size ))
				{
					printf("do_measurement:failed total i %d %d\n", i, total ) ;
					return ( 0 ) ;
				}

				total += usize ;
			}
		}
		frame_cnt++ ;
	}

}

/*
fin, fout : in out file descriptor
din_a     : add of input buffer on device
dout_a    : add of output buffer on device
hin_a     : add of host buffer for input from file
hout_a    : add of host buffer for output to file
ysize     : size of input for this comp in byte
dout_size : size of output buffer on device in byte
din_size  : size of input buffer on device in byte
*/

int
make_one_component( int fin, int fout,
	char *din_a, int *dout_a,
	char *hin_a, char *hout_a,
	int ysize, int dout_size, int din_size )
{
	int i ;

#ifdef CUDA_OBS 
	fprintf( stderr, "make_one_component: di %x do %x hi %x ho %x ins %d outs %d\n",
		din_a, dout_a, hin_a, hout_a, ysize, dout_size ) ;
#endif 

	if (( i = read ( fin, hin_a, ysize )) < 0 )
	{
		printf("make_one_component: read failed errno %d\n",
			errno ) ;
		return ( 0 ) ;
	}

	if ( !i )
		return ( 2 ) ;

	if ( i != ysize )
	{
		printf("make_one_component: read failed i %d\n", i ) ;
		return ( 0 ) ;
	}

	omp_timer_on( 0 ) ;

	if (( i = cudaMemcpy( din_a, hin_a, ysize, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		printf("make_one_component:download fail: %d\n", i ) ;
		return ( 0 ) ;
	}

	// dbg_p_d_data_c ( "after write", din_a, din_size ) ;

	cs_whm_measurement( din_a, ( int * )dout_a, din_size ) ; 

	if (( i = cudaMemcpy( hout_a, dout_a, dout_size, cudaMemcpyDeviceToHost))
		!= cudaSuccess )
	{
		printf("make_one_component:upload fail: %d\n", i ) ;
		return ( 0 ) ;
	}

	omp_timer_off( 0 ) ;

	// dbg_p_d_data_i ( "after measure",  dout_a, dout_size >> 2 ) ;

	if ( write ( fout, hout_a, dout_size ) != ( dout_size ))
	{
		printf("make_one_component: write failed errno %d\n",
			errno ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}
