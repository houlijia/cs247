#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

// #include "ibuf.h"
// #include "serial_wht3.h"
#include "cs_recon.h"
#include "tables.h"
#include "cs_cuda.h"

#include "cs_helper.h"
#include "cs_dbg.h"
#include "cs_buffer.h"
#include "cs_matrix.h"
#include "cs_perm_generic.h"
#include "cs_whm_encode_b.h"

// #define CUDA_DBG 

#define RGB_CNT		3

// parameter for worker thread

struct worker_thread_param {
	int *meap ;
	struct recon_param *pp ;
	int rgb_idx ;
	unsigned char *rgbp ;
} ;

int thread_ret[3] ;

// struct cs_buf_desc _cs_buf_desc ;

static int *d_sel_idxp = NULL, *sel_idxp = NULL ;

void div_h ( float *ffp, float d, int size ) ;
float *TV_GAP_rgb_use( int *ip, struct recon_param *para ) ;
void add ( float *ffp, float *ff1p, float *ttp, int size ) ;
void mul ( float *ffp, float d, int size ) ;
static float *h_cropping ( float *d_ip, int wht_size, int col, int row, int r_start, int c_start ) ;
static void h_a2d ( float *d_ip, unsigned char *d_tp, int offset, int size, float max ) ;
void *worker_thread( void *wp ) ;

#if 0
static float max_r, max_g, max_b ;
#endif
static unsigned char *rgbp ; // rgb image

// meap is the measuarements pointer
// outp is the rbg buffer, ready for display
// outsize is the length of the rbg buffer
unsigned char *
reconstruct( int *meap, struct recon_param *pp, int *outsize )
{
	pthread_t r_thread, g_thread, b_thread ;
	struct worker_thread_param r_param, g_param, b_param ;
	unsigned char *d_pixel_p ;
	int i, *r_ip, *g_ip, *b_ip, row, col ;

	dbg_init( 1024 * 1024 * 4 ) ;

	sel_idxp = selection_tbl[ pp->sel_idx ] ; 

	row = pp->r ;
	col = pp->c ;

	i = pp->wht_size ;

	i = i * i ;

#ifdef CUDA_OBS 
	k = buf_init( i * sizeof( float ), 8 ) ;

	if ( k == 0 )
	{
		printf("%s : buf_init failed \n", __func__ ) ;
		return ( NULL ) ;
	}
#endif 

#ifdef CUDA_OBS 
	_cs_buf_desc.size = sizeof ( float ) * i ;
	_cs_buf_desc.unit_size = i ;
	_cs_buf_desc.cnt = 10 ;

	if ( !cs_buffer_init ( &_cs_buf_desc, 1 ))
	{
		printf("%s : cs_buf_init failed \n", __func__ ) ;
		return ( NULL ) ;
	}
#endif 

	d_sel_idxp = ( int * ) cs_get_free_list ( 0 ) ;

	if ( d_sel_idxp == NULL )
	{
		printf("%s : d_sel_idxp failed \n", __func__ ) ;
		return ( NULL ) ;
	}

	if ( !dbg_put_d_data (( char *)d_sel_idxp, ( char *)sel_idxp, sizeof ( int ) * i ))
	{
		printf("%s : dbg_put_d_data failed \n", __func__ ) ;
		return ( NULL ) ;
	}

	h_do_scale_add_vector<int>( d_sel_idxp, -1, i ) ;	// 0 relative

	rgbp = ( unsigned char * ) malloc ( pp->r * pp->c * RGB_CNT ) ;

	if( rgbp == NULL )
	{
		printf("%s :rgbp  malloc failed\n", __func__ ) ;
		return ( NULL ) ;
	}

	p_buffer_dbg("init done") ;

	d_pixel_p = ( unsigned char * ) cs_get_free_list ( 0 ) ;

	if ( d_pixel_p == NULL )
	{
		printf("%s :d_pixel_p malloc failed\n", __func__ ) ;
		return ( NULL ) ;
	}

	r_param.meap = meap ;
	r_param.pp = pp ;
	r_param.rgbp = d_pixel_p ;
	r_param.rgb_idx = 2 ;

	i = pthread_create( &r_thread, NULL, worker_thread,( void *)&r_param ) ;
	if( i )
	{
		printf("%s - r_thread failed %d\n", __func__, i );
		return ( NULL ) ;
	}

	g_param.meap = meap + pp->sel_size ;
	g_param.pp = pp ;
	g_param.rgbp = d_pixel_p ;
	g_param.rgb_idx = 1 ;

	i = pthread_create( &g_thread, NULL, worker_thread,( void *)&g_param ) ;
	if( i )
	{
		printf("%s - g_thread failed %d\n", __func__, i );
		return ( NULL ) ;
	}

	b_param.meap = meap + pp->sel_size * 2 ;
	b_param.pp = pp ;
	b_param.rgbp = d_pixel_p ;
	b_param.rgb_idx = 0 ;

	i = pthread_create( &b_thread, NULL, worker_thread,( void *)&b_param ) ;
	if( i )
	{
		printf("%s - b_thread failed %d\n", __func__, i );
		return ( NULL ) ;
	}

	// error 

	pthread_join( r_thread, ( void ** ) &r_ip ) ;
	pthread_join( g_thread, ( void ** ) &g_ip ) ;
	pthread_join( b_thread, ( void ** ) &b_ip ) ;

	if (( *r_ip != 1 ) ||
		( *g_ip != 1 ) ||
		( *b_ip != 1 ))
	{
		cs_put_free_list(( char *) d_sel_idxp, 0 ) ;
		cs_put_free_list(( char *) d_pixel_p, 0 ) ;
		printf("thread done err r %d g %d b %d \n", *r_ip, *g_ip, *b_ip ) ;
		p_buffer_dbg("ALL DONE fail ...................................................") ;
		return ( NULL ) ;
	}


	p_buffer_dbg("COLOR loop done") ;

	cs_put_free_list(( char *) d_sel_idxp, 0 ) ;

	// squeeze ...

	*outsize = col * row * RGB_CNT ;

	printf("row %d col %d size %d \n", row, col, *outsize ) ;

	dbg_get_d_data(( char *) d_pixel_p, ( char *)rgbp, col * row * RGB_CNT ) ;

	cs_put_free_list(( char *) d_pixel_p, 0 ) ;

	p_buffer_dbg("ALL DONE ...................................................") ;

	return ( rgbp ) ;
}

__global__ static void
d_a2d ( float *d_fp, unsigned char *d_tp, int offset, int tbl_size, float max )
{
    float f, *fp ;
	unsigned char *cp ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		cp = d_tp + t_idx * RGB_CNT + offset  ;

		fp = d_fp + t_idx ;

		f = *fp ;

		if ( f > 0 )
		 	*cp = (( f / max ) * 255 + 0.5 ) ;
		else
			*cp = 0 ;	

		t_idx += CUDA_MAX_THREADS ;
	}
}

// when the return is not NULL, d_ip is freed ...
void
h_a2d ( float *d_ip, unsigned char *d_tp, int offset, int size, float max )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( size, nThreadsPerBlock, &nBlocks ) ;

	d_a2d <<< nBlocks, nThreadsPerBlock >>> ( d_ip, d_tp, offset, size, max ) ;

	cudaThreadSynchronize() ;
}

__global__ static void
d_cropping ( float *d_fp, float *d_tp, int tbl_size, int wht_size, int col, int row,
	int c_start, int r_start )
{
	int row_idx, col_idx ;
    float *fp, *tp ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		row_idx = t_idx / col ;
		col_idx = t_idx % col ;

		fp = d_fp + ( row_idx + r_start ) * wht_size + c_start + col_idx ; 
		tp = d_tp + t_idx ;

		*tp = *fp ;
	
		t_idx += CUDA_MAX_THREADS ;
	}
}

// when the return is not NULL, d_ip is freed ...
float *
h_cropping ( float *d_ip, int wht_size, int col, int row, int r_start, int c_start )
{
	int tbl_size ;
	float *tp ;
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	tbl_size = col * row ;

	if ( tbl_size <= 0 )
	{
		printf("%s : err col %d row %d \n", __func__, col, row ) ;
		return ( NULL ) ;
	}

	tp = ( float * )cs_get_free_list( 0 ) ;

	if ( tp == NULL )
	{
		printf("%s : err no buff\n", __func__ ) ;
		return ( NULL ) ;
	}

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	d_cropping <<< nBlocks, nThreadsPerBlock >>> ( d_ip, tp, tbl_size, wht_size, col,
		row, c_start, r_start ) ; 

	cudaThreadSynchronize() ;

	cs_put_free_list(( char *) d_ip, 0 ) ;

	return ( tp ) ;
}

void
i_recon_set_dbg( int *p )
{
	
	printf("set the idxp %p ", sel_idxp ) ;
    sel_idxp = p ;
	printf("TO idxp %p \n", sel_idxp ) ;
}	

// TV_GAP_rgb_use.m
// RGB ... one at a time

// the input data will be DESTROYED ... buffer WILL be returned
// input has the size of the wht_size * wht_size.  even if the valid
// data has only the idx_size number of elements

float *
A( float *ip, int wht_size, int *idxp, int idx_size )
{
	float *p2 ;
	int i ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "in A ...", ip, 8 * 8 ) ;
#endif 

	p2 = ( float * )cs_get_free_list( 0 ) ;

	if ( p2 == NULL )
	{
		printf("%s : p2 err \n", __func__ ) ;
		return ( NULL ) ;
	}

	i = wht_size * wht_size ;

	clear_device_mem_i(( int *) p2, i ) ;

	cs_whm_measurement_b<float> ( ip, i, i ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "in A after wht ", ip, i ) ;
#endif 

	// mea_select<float>( fp1, fp, idxp, idx_size ) ; 	// data is now in fp1
	h_do_permutation_generic_f2( ip, p2, idxp, idx_size ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "in A after select ", p2, i ) ;
#endif 

	cs_put_free_list(( char *) ip, 0 ) ;

	return ( p2 ) ;
}

// ip is on device
// data are in ip ... will not be changed, output is returned 
// T only supports float at this time ... 
template<typename T>
float *
At( T *ip, int wht_size, int *d_idxp, int idx_size )
{
	int size ;
	float *p1 ;

	p1 = ( float *)cs_get_free_list( 0 ) ;

	if ( p1 == NULL )
	{
		printf("%s : failed p1 %p\n", __func__, p1 ) ;
		return ( NULL ) ;
	}

	size = wht_size * wht_size ;

	// mea_un_select<T>( p1, ip, size, idxp, idx_size ) ; 

	h_do_permutation_generic_f1( ip, p1, d_idxp, size ) ;

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "input data", ( float *)ip, wht_size * wht_size ) ;
	dbg_p_d_data_f ( "now in float", p1, wht_size * wht_size ) ;
#endif 

	// data is in op ... ip is for buffer ... 
	// ip = wht_recon (( float *)ip, op, wht_size ) ;

	// fp = wht<float> ( p2, p1, wht_size ) ;

	cs_whm_measurement_b<float> ( p1, size, size ) ; 

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "after wht : p1", p1, wht_size * wht_size ) ;
#endif 

	// div_h ( fp, (float)(wht_size * wht_size ), size ) ;

	h_do_scale_mul_vector ( p1, 1.0/( float )size, size, p1 ) ; 

#ifdef CUDA_OBS 
	dbg_p_d_data_f ( "At done", p1, wht_size * wht_size ) ;
#endif 

	return ( p1 ) ;
}

// ip has the valid size of after the selection it is the r.g.b
// output is return ... which is the wht_size * wht_size matrix

float *
TV_GAP_rgb_use( int *ip, struct recon_param *para )
{
	int sel_idx, idx_size, wht_size ;
	int *d_ip, i, size, iter ;
	float *f, *tp, *fp, TVweight, lambda ;

	sel_idx = para->sel_idx ;
	idx_size = para->sel_size ;
	wht_size = para->wht_size ;

	size = wht_size * wht_size ;

	TVweight = para->TVweight ;

	lambda = para->lambda ;
	iter = para->iter ;

	printf("%s : sel_idx %d size %d wht %d tvweight %f lambda %f iter %d\n",
		__func__, sel_idx, idx_size, wht_size, TVweight, lambda, iter ) ;

	// mea_un_select<float, int>( op, ip, size, sel_idx, idx_size ) ; 

	d_ip = ( int *)cs_get_free_list( 0 ) ;

	if ( d_ip == NULL )
	{
		printf("%s : d_ip get buff err \n", __func__ ) ;
		return ( NULL ) ;
	}

	clear_device_mem_i (( int *) d_ip, size ) ;

	if ( !dbg_put_d_data (( char *) d_ip, ( char *)ip, sizeof ( int ) * idx_size ))
	{
		printf("%s : failed p1 %p ip %p\n", __func__, d_ip, ip ) ;
		return ( NULL ) ;
	}

	h_do_int_to_float ( d_ip,( float *)d_ip, idx_size ) ;	// p2 has the float data

	// d_ip has the data ... in float

	f = At<float> (( float *) d_ip, wht_size, d_sel_idxp, idx_size ) ;	// fp 2048 x 2048

	if ( f == NULL )
	{
		printf("%s : At err \n", __func__ ) ;
		return ( NULL ) ;
	}

	fp = ( float *)cs_get_free_list( 0 ) ;

	if ( fp == NULL )
	{
		printf("%s : fp buf_get failed \n", __func__ ) ;
		return ( 0 ) ;
	}

	// // memcpy ( fp, f, sizeof ( float ) * size ) ;

	for ( i = 0 ; i < iter ; i++ )
	{
		// memcpy ( fp, f, sizeof ( float ) * size ) ;

		h_do_copy_vector ( f, fp, size ) ;

#ifdef CUDA_OBS 
		printf("ITER ----------------------------------------------- %d \n", i ) ;
		dbg_p_d_data_f ( "in TV_GAP_rgb_use iter fp", fp, size) ;
		dbg_p_d_data_f ( "in TV_GAP_rgb_use iter f", f, size ) ;
		printf("ITER ----------------------------------------------- %d END \n", i ) ;
#endif 

		// fb        =   A( f_temp(:) );
		if (( fp = A( fp, wht_size, d_sel_idxp, idx_size )) == NULL )
		{
			printf("%s : A failed \n", __func__ ) ; // fp ... after selection
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after A", fp, size ) ;
		dbg_p_d_data_f ( "in d_ip ", ( float *)d_ip, size ) ;
#endif 

		// f(:,:,nr)         =   f(:,:,nr) + lambda.*reshape(At(( y(:,nr)-fb) ), [row, col]);
 
		// y(:,nr)-fb) 
#ifdef CUDA_OBS 
		tp = fp ;
		tip = ip ;
		j = idx_size ;
		while ( j-- )
		{
			*tp = (float )( *tip ) - *tp ;
	   		tp++ ;
			tip++ ;
		}	
#endif 
		h_do_vector_sub_vector(( float *) d_ip, fp, fp, idx_size ) ; 

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after sub", fp, size ) ;
#endif 

		// At(( y(:,nr)-fb)) ... op has the data
		if (( tp = At<float>( fp, wht_size, d_sel_idxp, idx_size )) == NULL )
		{
		  printf("%s:%d: At failed \n", __FILE__, __LINE__)	;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after At", tp, size ) ;
#endif 

		// lambda.*reshape(At(( y(:,nr)-fb) ), [row, col])
		h_do_scale_mul_vector( tp, lambda, size, fp ) ;

		cs_put_free_list(( char *) tp, 0 ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after mul", fp, size ) ;
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after mul", f, size ) ;
#endif 

		// f(:,:,nr) + lambda.*reshape(At(( y(:,nr)-fb) ), [row, col]);
		h_do_vector_add_vector ( fp, f, f, size ) ;

#ifdef CUDA_DBG 
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after add", f, size ) ;
#endif 

		// f(:,:,nr)          =   TV_denoising(f(:,:,nr),  TVweight,5);
		if(!TV_denoising( f, fp, TVweight, wht_size, wht_size, 5 ))
		{
			printf("%s: TV_denoising failed \n", __func__ ) ;
			return ( NULL ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after TV_denoising f ", f, size ) ;
		dbg_p_d_data_f ( "in TV_GAP_rgb_use after TV_denoising fp ", fp, size ) ;
#endif 
		cs_buf_swap ( &f, &fp ) ; // good data is in f

#ifdef CUDA_DBG 
		p_buffer_dbg("after iter ...................................") ;
#endif 

#ifdef CUDA_DBG 
		dbg_p_d_data_f ( "LDL ... TV_denoising f ", f, size ) ;
#endif 

	}

	cs_put_free_list(( char *) fp, 0 ) ;	// good data is in f
	cs_put_free_list(( char *) d_ip, 0 ) ;	// good data is in f

	p_buffer_dbg("done with TV_GAP_rgb_use") ; 

	return ( f ) ;
}	

// TV_denoising.m

// do_t means call dvt

__global__ void
d_dv ( float *d_fp, float *d_tp, int col, int row, int tbl_size )
{
	int row_idx, col_idx ;
    float *fp, f, *tp ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		row_idx = t_idx / col ;
		col_idx = t_idx % col ;

		tp = d_tp + t_idx ;
		fp = d_fp + row_idx * col + col_idx ;

		f = *fp ;
		fp += col ;

		*tp = *fp - f ;
	
		t_idx += CUDA_MAX_THREADS ;
	}
}

__global__ void
d_dvt ( float *d_fp, float *d_tp, int col, int row, int tbl_size )
{
	int row_idx, col_idx ;
    float f, *fp, *tp ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;

	while ( t_idx < tbl_size )
	{
		row_idx = t_idx / col ;
		col_idx = t_idx % col ;

		tp = d_tp + t_idx ;
		fp = d_fp + row_idx * col + col_idx ;

		if ( row_idx == 0 )
			*tp = -*fp ;
		else 
		{
			fp -= col ;
			if ( row_idx == row )
			{
				*tp = *fp ;
			} else
			{
				f = *fp ;
				fp += col ;
	
				*tp = f - *fp ;
			}
		}
	
		t_idx += CUDA_MAX_THREADS ;
	}
}

// input 2047x2048 ... if no do_t, output 2046x2048
// if with do_t, output 2048*2048 
// do_t means call dvt()
int 
h_dv( float *ffp, float *ttp, int col, int row, int do_t )
{
	int tbl_size ;
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	if ( do_t )
		tbl_size = ( row + 1 ) * col ;
	else
		tbl_size = ( row - 1 ) * col ;

	if ( tbl_size <= 0 )
	{
		printf("%s : err col %d row %d do_t %d \n", __func__, col, row, do_t ) ;
		return ( 0 ) ;
	}

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	if ( do_t )
		d_dvt <<< nBlocks, nThreadsPerBlock >>> ( ffp, ttp, col, row, tbl_size) ; 
	else
		d_dv <<< nBlocks, nThreadsPerBlock >>> ( ffp, ttp, col, row, tbl_size) ; 

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

// do_t is 0
__global__ void
d_dh ( float *d_fp, float *d_tp, int col, int row, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int ncol, row_idx, col_idx ;
	float f, *fp, *tp ;

	ncol = col - 1 ;

	while ( t_idx < tbl_size )
	{
		row_idx = t_idx / ncol ;
		col_idx = t_idx % ncol ;

		tp = d_tp + t_idx ;

		fp = d_fp + row_idx * col + col_idx ;

		f = *fp++ ;
		*tp = *fp - f ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

// do_t is 1
__global__ void
d_dht ( float *d_fp, float *d_tp, int col, int row, int tbl_size )
{
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int ncol, row_idx, col_idx ;
	float *fp, f, *tp ;

	ncol = col + 1 ;

	while ( t_idx < tbl_size )
	{
		row_idx = t_idx / ncol ;
		col_idx = t_idx % ncol ;

		tp = d_tp + t_idx ;

		fp = d_fp + row_idx * col + col_idx ;

		if ( col_idx == 0 )
			*tp = -*fp ;
		else if ( col_idx == ( ncol - 1 ))
		{
			fp-- ;
			*tp = *fp ;
		}
		else
		{
			f = *fp-- ;
			*tp = *fp - f ;
		}

		t_idx += CUDA_MAX_THREADS ;
	}
}

// input 2048x2047 ... if no do_t, output 2048x2046
// if with do_t, output 2048*2048 
// do_t means call dht()
int 
h_dh( float *ffp, float *ttp, int col, int row, int do_t )
{
	int tbl_size ;
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	if ( do_t )
		tbl_size = row * ( col + 1 ) ;
	else
		tbl_size = row * ( col - 1 ) ;

	if ( tbl_size <= 0 )
	{
		printf("%s : err col %d row %d do_t %d \n", __func__, col, row, do_t ) ;
		return ( 0 ) ;
	}

	h_block_adj ( tbl_size, nThreadsPerBlock, &nBlocks ) ;

	if ( do_t )
		d_dht <<< nBlocks, nThreadsPerBlock >>> ( ffp, ttp, col, row, tbl_size) ; 
	else
		d_dh <<< nBlocks, nThreadsPerBlock >>> ( ffp, ttp, col, row, tbl_size) ; 

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

#ifdef CUDA_OBS 
// the output has one extra row ...

int 
dvt ( float *ffp, float *ttp, int col, int row )
{
	int i ;
	
	i = dv( ffp, ttp, col, row, 1 ) ;

	return ( i ) ;
}

// out put has extra col

int 
dht ( float *ffp, float *ttp, int col, int row )
{
	int i ;

	i = dh ( ffp, ttp, col, row, 1 ) ;

	return ( i ) ;
}
#endif 

int
clip ( float *ffp, float *ttp, int size, float lambda )
{
	float v, vv, vvv ;

	while ( size-- )
	{
		vvv = *ffp++ ;

		v = ( vvv > 0 ) ? vvv : -vvv ;

		vv = ( v > lambda )? lambda : v ;

		// printf("vvv %f vv %f v %f lambda %f \n", vvv, vv, v, lambda ) ;

		if ( vvv > 0 )
			*ttp++ = vv ;
		else
			*ttp++ = -vv ;
	}
	return ( 1 ) ;
}

void
sub ( float *ffp, float *ff1p, float *ttp, int size )
{
#ifdef CUDA_DBG 
	printf("%s : f %p f1 %p tp %p\n", __func__, ffp, ff1p, ttp ) ;
	dbg_p_d_data_f("ffp", ffp, 8 * 8 ) ;
	dbg_p_d_data_f("ff1p", ff1p, 8 * 8 ) ;
	dbg_p_d_data_f("ttp", ttp, 8 * 8 ) ;
#endif 

	while ( size-- )
		*ttp++ = *ffp++ - *ff1p++ ;
}

void
add ( float *ffp, float *ff1p, float *ttp, int size )
{
	while ( size-- )
		*ttp++ = *ffp++ + *ff1p++ ;
}
			
void
bdiv ( float *ffp, float d, int size )
{
	float d1 ;

	while ( size-- )
	{
		d1 = *ffp ;
		*ffp++ = d / d1 ;
	}
}
			
void
mul ( float *ffp, float d, int size )
{
	while ( size-- )
		*ffp++ *= d ;
}
			
void
div_h ( float *ffp, float d, int size )
{
	while ( size-- )
		*ffp++ /= d ;
}

__global__ void
d_clip ( float *d_fp, float *d_tp, int tbl_size, float lambda )
{
    float *fp, *tp ;
	int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
	float vv, v, vvv ;

	while ( t_idx < tbl_size )
	{
		fp = d_fp + t_idx ;
		tp = d_tp + t_idx ;

		vvv = *fp ;

		v = ( vvv > 0 ) ? vvv : -vvv ;

		vv = ( v > lambda )? lambda : v ;

		if ( vvv > 0 )
			*tp++ = vv ;
		else
			*tp++ = -vv ;

		t_idx += CUDA_MAX_THREADS ;
	}
}

void
h_clip ( float *d_inp, float *d_outp, int size, float lambda )
{
	int nThreadsPerBlock = CUDA_MAX_THREADS_P_BLK ;
	int nBlocks ; // = ( tbl_size + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	h_block_adj ( size, nThreadsPerBlock, &nBlocks ) ;

	d_clip <<< nBlocks, nThreadsPerBlock >>> ( d_inp, d_outp, size, lambda ) ; 

	cudaThreadSynchronize() ;
}

// *y0 size wht * wht ... assume col 2048 row 2048
int
TV_denoising( float *y0, float *x0, float lambda, int col, int row, int iter )
{
	int i ;
	float alpha ;
	float *v0h = NULL, *v0v = NULL, *zh = NULL, *zv = NULL ;

	alpha = 5 ;

	// y0 size is 2048 * 2048

#ifdef CUDA_DBG 
	printf("%s ::: lambda %f col %d row %d iter %d alpha %f\n",
		__func__, lambda, col, row, iter, alpha ) ;
#endif 
	
	if ( row == 1 )
	{
		printf("%s : row is one %d ... need to implement \n", __func__, row ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_DBG 
	p_buffer_dbg("before denoising --------------------------------------------------------------\n") ;
	// dbg_p_d_data_f("input ", y0, col * row ) ;
#endif 

	clear_device_mem_i (( int *) x0, col * row ) ;		// z 2048 * 2047

	// double check the size changes ... LDL 

	zh = ( float *)cs_get_free_list (0) ;
	zv = ( float *)cs_get_free_list(0) ;
	v0h = ( float *)cs_get_free_list(0) ;
	v0v = ( float *)cs_get_free_list(0) ;

	if (( zh == NULL ) ||
		( zv == NULL ) ||
		( v0h == NULL ) ||
		( v0v == NULL ))
	{
		printf("%s : err buf %p %p %p %p \n", __func__, zh, zv, v0h, v0v ) ;
		p_buffer_dbg("TV_denoising") ;
		return ( 0 ) ;
	}

	clear_device_mem_i (( int *) zh, col * row ) ;	// 2047 * 2048
	clear_device_mem_i (( int *) zv, col * row ) ;	// 2048 * 2047

	clear_device_mem_i (( int *) v0h, col * row ) ;	// 2048 * 2048
	clear_device_mem_i (( int *) v0v, col * row ) ;	// 2048 * 2048

#ifdef CUDA_DBG 
	printf("alpha %f lambda %f \n", alpha, lambda ) ;
#endif 

	for ( i = 0 ; i < iter ; i++ )
	{
		// v0h = y0 - dht(zh);

#ifdef CUDA_OBS
		printf("ITER --------------------------------------------------------------- %d \n", i ) ;
#endif 

		if ( !h_dh( zh, v0h, col - 1, row, 1 ))
		{
			printf("%s : dht return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0h ------ after dht", v0h, col * row ) ;
#endif 

		// sub ( y0, v0h, v0h, col * row ) ;
		h_do_vector_sub_vector ( y0, v0h, v0h, col * row ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0h ------ after sub ", v0h, col * row ) ;
#endif 

		// v0v = y0 - dvt(zv);

		if ( !h_dv( zv, v0v, col, row - 1, 1 ))
		{
			printf("%s : dvt return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "LDL v0v ------ after dvt", v0v, col * row ) ;
#endif 

		// sub ( y0, v0v, v0v, col * row ) ;
		h_do_vector_sub_vector ( y0, v0v, v0v, col * row ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "LDL v0v ------ after sub ", v0v, col * row ) ;
#endif 

		// x0 = (v0h + v0v)./2;

		// add ( v0h, v0v, x0, col * row ) ;
		h_do_vector_add_vector( v0h, v0v, x0, col * row ) ;

		// div_h ( x0, 2, col * row ) ;

		h_do_scale_mul_vector( x0, 0.5, col * row, x0 ) ;

#ifdef CUDA_OBS
		dbg_p_d_data_f ( "x0 ------ after div_h ", x0, col * row ) ;
#endif 

		if ( i == ( iter - 1 ))
			break ;

		// zh = clip(zh + 1/alpha*dh(x0), lambda/2);

		if ( !h_dh ( x0, v0h, col, row, 0 ))
		{
			printf("%s : dh return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0h ------ after dh ", v0h, ( col - 1 ) * row ) ;
#endif 

		// mul ( v0h, 1/alpha, ( col - 1 ) * row ) ;
		h_do_scale_mul_vector ( v0h, 1/alpha, ( col - 1 ) * row, v0h ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0h ------ after mul ", v0h, ( col - 1 ) * row ) ;
#endif 

		// add ( v0h, zh, v0h, ( col - 1 ) * row ) ;
		h_do_vector_add_vector ( v0h, zh, v0h, ( col - 1 ) * row ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0h ------ after add ", v0h, ( col - 1 ) * row ) ;
#endif 

		h_clip ( v0h, zh, col * row, lambda / 2 ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "zh ------ after clip", zh, ( col - 1 ) * row ) ;
#endif 

		// zv = clip(zv + 1/alpha*dv(x0), lambda/2);

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "LDL ------ before h_dv x0", x0, col * row ) ;
		dbg_p_d_data_f ( "LDL ------ before h_dv v0v", v0v, col * row ) ;
#endif 

		if ( !h_dv ( x0, v0v, col, row, 0 ))
		{
			printf("%s : dv return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0v ------ after dv ", v0v, col *( row - 1 )) ;
#endif 

		// mul ( v0v, 1/alpha, col * ( row - 1 )) ;
		h_do_scale_mul_vector ( v0v, 1/alpha, col * ( row - 1 ), v0v) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0v ------ after mul ", v0v, col * ( row - 1 )) ;
#endif 

		// add ( v0v, zv, v0v, col * ( row - 1 )) ;
		h_do_vector_add_vector ( v0v, zv, v0v, col * ( row - 1 )) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "v0v ------ after add ", v0v, col * ( row - 1 )) ;
#endif 

		// clip ( v0v, zv, col * row, lambda / 2 ) ;
		h_clip ( v0v, zv, col * row, lambda / 2 ) ;

#ifdef CUDA_OBS 
		dbg_p_d_data_f ( "zv ------ after clip", zv, col * ( row -1 )) ;
#endif 
	}

	cs_put_free_list(( char *) zh, 0 ) ;
	cs_put_free_list(( char *) zv, 0 ) ;
	cs_put_free_list(( char *) v0h, 0 ) ;
	cs_put_free_list(( char *) v0v, 0 ) ;

	p_buffer_dbg("recon done") ;

	return ( 1 ) ;
}

void *
worker_thread( void *wwp )
{
	struct worker_thread_param *wp ;
	int *meap ;
	struct recon_param *pp ;
	int rgb_idx ;
	unsigned char *rgbp ;
	float max_r, *rp, *fp ;
	int i ;

	wp = ( struct worker_thread_param * )wwp ;

	meap = wp->meap ;
	pp = wp->pp ;
	rgb_idx = wp->rgb_idx ;
	rgbp = wp->rgbp ;

	if (( fp = TV_GAP_rgb_use( meap, pp )) == NULL )
	{
		printf("%s : TV_GAP_rgb_use failed idx %d \n", __func__, rgb_idx ) ;
		thread_ret[ rgb_idx ] = 1111 ;
		pthread_exit (( void * ) &thread_ret[ rgb_idx ] ) ;
	}

	rp = h_cropping ( fp, pp->wht_size, pp->c, pp->r, pp->r_start, pp->c_start ) ;

	if ( rp == NULL )
	{ 
		printf("%s : %d fp cropping failed \n", __func__, rgb_idx ) ;
		cs_put_free_list(( char *) fp, 0 ) ;
		thread_ret[ rgb_idx ] = 111 ;
		pthread_exit (( void * ) &thread_ret[ rgb_idx ] ) ;
	}

	i = pp->r * pp->c ;

	fp = ( float * ) cs_get_free_list ( 0 ) ;

	if (!dbg_copy_d_data(( char *) fp, ( char *)rp, sizeof ( float ) * i ))
	{
		printf("%s : bp cropping failed \n", __func__ ) ;
		cs_put_free_list(( char *) fp, 0 ) ;
		thread_ret[ rgb_idx ] = 11 ;
		pthread_exit (( void * ) &thread_ret[ rgb_idx ] ) ;
	}

	max_r = h_do_max_destroy ( fp, i ) ;

	h_a2d ( rp, ( unsigned char *)rgbp, rgb_idx, i, max_r ) ;

	cs_put_free_list(( char *) rp, 0 ) ;
	cs_put_free_list(( char *) fp, 0 ) ;

	thread_ret[ rgb_idx ] = 1 ;
	pthread_exit (( void * ) &thread_ret[ rgb_idx ] ) ;
}
