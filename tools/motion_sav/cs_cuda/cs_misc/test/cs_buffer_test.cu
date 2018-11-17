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
#include "cs_buffer.h"

#define CUDA_DBG
#define BUF_SIZE_INT		10

float *fp1 ;

struct cs_buf_desc cbd[] = {
	{ 1000, 2 },
	{ 2000, 3 },
	{ 3000, 4 }
} ;

int
main( int ac, char *av[] )
{
	char *p1, *p2, *p3, *p4 ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	dbg_init ( 1024 * 1024 ) ;

	cs_buffer_init ( cbd, 3 ) ;

	p_buffer_dbg("after init") ;

	printf("get 2 from 0 -----------------------------------------------------\n") ;

	p1 = cs_get_free_list ( 0 ) ;
	p2 = cs_get_free_list ( 0 ) ;

	printf("get 2 from 0: p1 %p p2 %p \n", p1, p2 ) ;

	p_buffer_dbg("get 2 from 0") ;

	cs_put_free_list( p2, 0 ) ;
	cs_put_free_list( p1, 0 ) ;

	p_buffer_dbg("put 2 from 0") ;

	printf("get from 1/2 -----------------------------------------------------\n") ;

	p1 = cs_get_free_list ( 1 ) ;
	p2 = cs_get_free_list ( 2 ) ;

	printf("get from 1/2: p1 %p p2 %p \n", p1, p2 ) ;

	p_buffer_dbg("get from 1/2") ;

	cs_put_free_list( p1, 1 ) ;
	cs_put_free_list( p2, 2 ) ;

	p_buffer_dbg("put to 1/2") ;

	printf("get 4 from 1 -----------------------------------------------------\n") ;

	p1 = cs_get_free_list ( 1 ) ;
	p2 = cs_get_free_list ( 1 ) ;
	p3 = cs_get_free_list ( 1 ) ;
	p4 = cs_get_free_list ( 1 ) ;

	printf("get 4 from 1: p1 %p p2 %p p3 %p p4 %p \n", p1, p2, p3, p4 ) ;

	p_buffer_dbg("get from 1/2") ;

	if ( p1 )
		cs_put_free_list( p2, 1 ) ;

	if ( p2 ) 
		cs_put_free_list( p3, 1 ) ;

	if ( p3 )
		cs_put_free_list( p1, 1 ) ;

	if ( p4 )
		cs_put_free_list( p4, 1 ) ;

	p_buffer_dbg("put 4 to 1") ;
}
