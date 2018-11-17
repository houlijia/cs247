#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "ibuf.h"
#include "serial_wht3.h"
#include "i_recon.h"
#include "tables.h"

// #define CUDA_DBG 

#define RGB_CNT		3

int *sel_idxp = NULL ;

void div_h ( float *ffp, float d, int size ) ;
float *TV_GAP_rgb_use( int *ip, struct recon_param *para ) ;
void add ( float *ffp, float *ff1p, float *ttp, int size ) ;
void mul ( float *ffp, float d, int size ) ;

float max_r, max_g, max_b ;
unsigned char *rgbp ; // rgb image

// meap is the measuarements pointer
// outp is the rbg buffer, ready for display
// outsize is the length of the rbg buffer
unsigned char *
reconstruct( int *meap, struct recon_param *pp, int *outsize )
{
	int i, j, k, row, col ;
	unsigned char *cp ;
	float f, *rp=NULL, *gp=NULL, *bp=NULL, *rpf, *rpt, *gpf, *gpt, *bpf, *bpt, *fp ;

	sel_idxp = selection_tbl[ pp->sel_idx ] ; 

	row = pp->r ;
	col = pp->c ;

	i = pp->wht_size ;

	i = i * i ;

	k = buf_init( i * sizeof( float ), 8 ) ;

	if ( k == 0 )
	{
		printf("%s : buf_init failed \n", __func__ ) ;
		return ( NULL ) ;
	}

	rgbp = ( unsigned char * ) malloc ( pp->r * pp->c * RGB_CNT ) ;

	if( rgbp == NULL )
	{
		printf("%s :rgbp  malloc failed\n", __func__ ) ;
		return ( NULL ) ;
	}

	buf_p("init done") ;

	for ( i = 0 ; i < RGB_CNT ; i++ )
	{
		if (( fp = TV_GAP_rgb_use( meap, pp )) == NULL )
		{
			printf("%s : TV_GAP_rgb_use failed i %d \n", __func__, i ) ;
			return ( NULL ) ;
		}

		switch ( i ) {
		case 0 :
#ifdef CUDA_DBG 
			p_num_nm_f("RRRRR", fp, 8, 8 ) ;
#endif 
			rp = fp ;
			break ;

		case 1 :
#ifdef CUDA_DBG 
			p_num_nm_f("GGGGG", fp, 8, 8 ) ;
#endif 
			gp = fp ;
			break ;

		case 2 :
#ifdef CUDA_DBG 
			p_num_nm_f("BBBBB", fp, 8, 8 ) ;
#endif 
			bp = fp ;
			break ;
		default:
		  assert(false);
		}

		buf_p("after TV_GAP swap buf") ;

		meap += pp->sel_size ;	// RGB ...
	}

	// find the max of r.g.b. ... and squeeze ...

	j = ( pp->r_start * pp->wht_size ) + pp->c_start ; 
	rpf = rp + j ;
	gpf = gp + j ;
	bpf = bp + j ;

	rpt = rp ;
	gpt = gp ;
	bpt = bp ;

	max_r = max_g = max_b = 0 ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
		{
			// R

			f = *rpf++ ;
			if ( f > max_r )
				max_r = f ;
			*rpt++ = f ;

			// G

			f = *gpf++ ;
			if ( f > max_g )
				max_g = f ;
			*gpt++ = f ;

			// B

			f = *bpf++ ;
			if ( f > max_b )
				max_b = f ;
			*bpt++ = f ;
		}

		j = pp->wht_size - col ;
		rpf += j ;
		gpf += j ;
		bpf += j ;
	}

#ifdef CUDA_OBS 
	p_num_nm_f ("RRR--", rp, pp->c, pp->r );
	p_num_nm_f ("GGG--", gp, pp->c, pp->r );
	p_num_nm_f ("BBB--", bp, pp->c, pp->r );
#endif 

	printf("MAX rrr %f ggg %f bbb %f \n", max_r, max_g, max_b ) ;

	// get the pixel value ... in rgb format

	rpf = rp ;
	gpf = gp ;
	bpf = bp ;

	cp = rgbp ;

	i = col * row ;
	while ( i-- )
	{
		f = *bpf++ ;
		if ( f > 0 )
			*cp++ = ( unsigned char ) ((( f / max_b ) * 255 ) + 0.5 ) ; 
		else
			*cp++ = 0 ;

		f = *gpf++ ;
		if ( f > 0 )
			*cp++ = ( unsigned char ) ((( f / max_g ) * 255 ) + 0.5 ) ; 
		else
			*cp++ = 0 ;

		f = *rpf++ ;
		if ( f > 0 )
			*cp++ = ( unsigned char ) ((( f / max_r ) * 255 ) + 0.5 ) ; 
		else 
			*cp++ = 0 ;
	}

	*outsize = col * row * RGB_CNT ;

	printf("row %d col %d size %d \n", row, col, *outsize ) ;

	buf_put(( char *) rp ) ;
	buf_put(( char *) gp ) ;
	buf_put(( char *) bp ) ;

	buf_p("ALL DONE ...................................................") ;

	return ( rgbp ) ;
}

void
i_recon_set_dbg( int *p )
{
	
  printf("set the idxp %p ", (void*)sel_idxp ) ;
    sel_idxp = p ;
    printf("TO idxp %p \n", (void*)sel_idxp ) ;
}	

// TV_GAP_rgb_use.m
// RGB ... one at a time

// ip could potentially buf_free()'ed ... but either way should not be used ...
// the input data will be DESTROYED
// input has the size of the wht_size * wht_size.  even if the valid
// data has only the idx_size number of elements

float *
A( float *ip, int wht_size, int *idxp, int idx_size )
{
	float *p1, *p2, *fp, *fp1 ;

	p1 = ip ;
	p2 = ( float *)buf_get() ;

	if ( p2 == NULL )
	{
		printf("%s: err p2 \n", __func__ ) ;
		return ( 0 ) ;
	}

#ifdef CUDA_OBS 
	p_num_nm_f ( "in A ...", ip, 8, 8 ) ;
#endif 

	fp = wht<float> ( p2, p1, wht_size ) ;

#ifdef CUDA_OBS 
	p_num_nm_f ( "in A after wht ", ip, 8, 8 ) ;
#endif 

	if ( fp == p1 )
		fp1 = p2 ;
	else
		fp1 = p1 ;

	memset ( fp1, 0, wht_size * wht_size * sizeof ( float ) ) ;

	mea_select<float>( fp1, fp, idxp, idx_size ) ; 	// data is now in fp1

#ifdef CUDA_OBS 
	p_num_nm_f ( "in A after select ", fp1, 1, wht_size * wht_size ) ;
#endif 

	buf_put(( char *) fp ) ;

	return ( fp1 ) ;
}

// data are in ip ... will not be changed, output is returned 
template<typename T>
float *
At( T *ip, int wht_size, int *idxp, int idx_size )
{
	int size ;
	float *p1, *p2, *fp ;

	p1 = ( float *)buf_get() ;
	p2 = ( float *)buf_get() ;

	if (( p1 == NULL ) || ( p2 == NULL ))
	{
	  printf("%s : failed p1 %p p2 %p\n", __func__, (void*)p1, (void*)p2 ) ;
		return ( NULL ) ;
	}

	size = wht_size * wht_size ;

	mea_un_select<T>( p1, ip, size, idxp, idx_size ) ; 

#ifdef CUDA_OBS 
	p_num_nm_x<T> ( "input data", ip, 1, idx_size ) ;
	p_num_nm_f ( "now in float", p1, wht_size, wht_size ) ;
#endif 

	// data is in op ... ip is for buffer ... 
	// ip = wht_recon (( float *)ip, op, wht_size ) ;

	fp = wht<float> ( p2, p1, wht_size ) ;

#ifdef CUDA_OBS 
	p_num_nm_f ( "after wht", fp, wht_size, wht_size ) ;

#ifdef CUDA_OBS 
	if ( fp != p2 )
		p_num_nm_f ( "after wht : p2", p2, wht_size, wht_size ) ;
	else
		p_num_nm_f ( "after wht : p1", p1, wht_size, wht_size ) ;
#endif 
#endif 

	div_h ( fp, (float)(wht_size * wht_size ), size ) ;

	if ( fp == p1 )
		buf_put(( char *) p2 ) ;
	else
		buf_put(( char *) p1 ) ;

#ifdef CUDA_OBS 
	p_num_nm_f ( "At done", fp, wht_size, wht_size ) ;
#endif 

	return ( fp ) ;
}

// ip has the valid size of after the selection it is the r.g.b
// output is return ... which is the wht_size * wht_size matrix

float *
TV_GAP_rgb_use( int *ip, struct recon_param *para )
{
	int sel_idx, idx_size, wht_size ;
	int *tip, i, j, size, iter ;
	//	int row, col;
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

	// f has the data ...

	f = At<int> ( ip, wht_size, sel_idxp, idx_size ) ;	// fp 2048 x 2048

	if ( f == NULL )
	{
		printf("%s : At err \n", __func__ ) ;
		return ( NULL ) ;
	}

	fp = ( float *)buf_get() ;

	if ( fp == NULL )
	{
		printf("%s : fp buf_get failed \n", __func__ ) ;
		return ( 0 ) ;
	}

	// // memcpy ( fp, f, sizeof ( float ) * size ) ;

	for ( i = 0 ; i < iter ; i++ )
	{
		memcpy ( fp, f, sizeof ( float ) * size ) ;

#ifdef CUDA_OBS 
		printf("ITER ----------------------------------------------- %d \n", i ) ;
		p_num_nm_f ( "in TV_GAP_rgb_use iter fp", fp, 1, size) ;
		p_num_nm_f ( "in TV_GAP_rgb_use iter f", f, 1, size ) ;
		printf("ITER ----------------------------------------------- %d END \n", i ) ;
#endif 

		// fb        =   A( f_temp(:) );
		if (( fp = A( fp, wht_size, sel_idxp, idx_size )) == NULL )
		{
			printf("%s : A failed \n", __func__ ) ; // fp ... after selection
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "in TV_GAP_rgb_use after A", fp, 1, size ) ;
		p_num_nm ( "in ip ", ip, 1, idx_size ) ;
#endif 

		// f(:,:,nr)         =   f(:,:,nr) + lambda.*reshape(At(( y(:,nr)-fb) ), [row, col]);
 
		// y(:,nr)-fb) 
		tp = fp ;
		tip = ip ;
		j = idx_size ;
		while ( j-- )
		{
			*tp = (float )( *tip ) - *tp ;
	   		tp++ ;
			tip++ ;
		}	

#ifdef CUDA_OBS 
		p_num_nm_f ( "in TV_GAP_rgb_use after sub", fp, 1, size ) ;
#endif 

		// At(( y(:,nr)-fb)) ... op has the data
		if (( tp = At<float>( fp, wht_size, sel_idxp, idx_size )) == NULL )
		{
		  printf("%s : At failed \n",__func__)	;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "in TV_GAP_rgb_use after At", tp, 1, size ) ;
#endif 

		buf_put(( char *) fp ) ;
		fp = tp ;

		// lambda.*reshape(At(( y(:,nr)-fb) ), [row, col])
		mul( fp, lambda, size ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "in TV_GAP_rgb_use after mul", fp, 1, size ) ;
		p_num_nm_f ( "in TV_GAP_rgb_use after mul", f, 1, size ) ;
#endif 

		// f(:,:,nr) + lambda.*reshape(At(( y(:,nr)-fb) ), [row, col]);
		add ( fp, f, f, size ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "in TV_GAP_rgb_use after add", f, 1, size ) ;
#endif 

		// f(:,:,nr)          =   TV_denoising(f(:,:,nr),  TVweight,5);
		if ( !TV_denoising( f, fp, TVweight, wht_size, wht_size, 5 ))
		{
			printf("%s: TV_denoising failed \n", __func__ ) ;
			return ( NULL ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "in TV_GAP_rgb_use after TV_denoising f ", f, 1, size ) ;
		p_num_nm_f ( "in TV_GAP_rgb_use after TV_denoising fp ", fp, 1, size ) ;
#endif 
		buf_swap ( &f, &fp ) ; // good data is in f

#ifdef CUDA_OBS 
		buf_p("after iter ...................................") ;
#endif 

#ifdef CUDA_DBG 
		p_num_nm_f ( "LDL ... TV_denoising f ", f, 1, size ) ;
#endif 

	}

	buf_put(( char *) fp ) ;	// good data is in f

	return ( f ) ;
}	

// TV_denoising.m

// do_t means call dvt

int
dv( float *ffp, float *ttp, int col, int row, int do_t )
{
	int i, j ;
    float *fp1, *fp2;

	if ( row == 1 )
	{
		printf("%s :: err row is 1 \n", __func__ ) ;
		return ( 0 ) ;
	}

	if ( do_t )
	{
		fp1 = ffp ;
		for ( i = 0 ; i < col ; i++ )
			*ttp++ = -*fp1++ ;
	}

	fp1 = ffp ;
	fp2 = ffp + col ;
	for ( i = 0 ; i < row - 1 ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
		{
			if ( do_t )
				*ttp++ = *fp1++ - *fp2++ ;
			else
				*ttp++ = *fp2++ - *fp1++ ;
		}
	}

	if ( do_t )
	{
		fp1 = ffp + ( row - 1 ) * col ;
		for ( i = 0 ; i < col ; i++ )
			*ttp++ = *fp1++ ;
	}

	return ( 1 ) ;
}

// do_t means call dht()
int 
dh( float *ffp, float *ttp, int col, int row, int do_t )
{
	int i, j ;
	float *fp1, *fp2;

	if ( col == 1 )
	{
		printf("%s :: err col is 1 \n", __func__ ) ;
		return ( 0 ) ;
	}

	for ( i = 0 ; i < row ; i++ )
	{
		fp1 = ffp + i * col ;
		fp2 = fp1 + 1 ;

		if ( do_t ) 
			*ttp++ = -*fp1 ;

		for ( j = 0 ; j < col - 1 ; j++ )
		{
			if ( do_t )
				*ttp++ = *fp1++ - *fp2++ ; // negate ...
			else
				*ttp++ = *fp2++ - *fp1++ ; 
		}

		if ( do_t )
			*ttp++ = *fp1 ;
	}
	return ( 1 ) ;
}

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
	p_num_nm_x<float>("ffp", ffp, 8, 8 ) ;
	p_num_nm_x<float>("ff1p", ff1p, 8, 8 ) ;
	p_num_nm_x<float>("ttp", ttp, 8, 8 ) ;
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
			
// *y0 size wht * wht ... assume col 2048 row 2048
int
TV_denoising( float *y0, float *x0, float lambda, int col, int row, int iter )
{
	int i;
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
	buf_p("before denoising --------------------------------------------------------------\n") ;
	p_num_nm_x<float>("input ", y0, col, row ) ;
#endif 

	memset ( x0, 0, sizeof ( float ) * col * row ) ;		// z 2048 * 2047

	// double check the size changes ... LDL 

	zh = ( float *)buf_get() ;
	zv = ( float *)buf_get() ;
	v0h = ( float *)buf_get() ;
	v0v = ( float *)buf_get() ;

	if (( zh == NULL ) ||
		( zv == NULL ) ||
		( v0h == NULL ) ||
		( v0v == NULL ))
	{
	  printf("%s : err buf %p %p %p %p \n", __func__, (void*)zh, (void*)zv, (void*)v0h, (void*)v0v ) ;
		buf_p("TV_denoising") ;
		return ( 0 ) ;
	}

	memset ( zh, 0, sizeof ( float ) * col * row ) ;	// 2047 * 2048
	memset ( zv, 0, sizeof ( float ) * col * row ) ;	// 2048 * 2047

	memset ( v0h, 0, sizeof ( float ) * col * row ) ;	// 2048 * 2048
	memset ( v0v, 0, sizeof ( float ) * col * row ) ;	// 2048 * 2048

#ifdef CUDA_DBG 
	printf("alpha %f lambda %f \n", alpha, lambda ) ;
#endif 

	for ( i = 0 ; i < iter ; i++ )
	{
		// v0h = y0 - dht(zh);

#ifdef CUDA_OBS 
		printf("ITER --------------------------------------------------------------- %d \n", i ) ;
#endif 

		if ( !dht( zh, v0h, col - 1, row ))
		{
			printf("%s : dht return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0h ------ after dht", v0h, col, row ) ;
#endif 

		sub ( y0, v0h, v0h, col * row ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0h ------ after sub ", v0h, col, row ) ;
#endif 

		// v0v = y0 - dvt(zv);

		if ( !dvt( zv, v0v, col, row - 1 ))
		{
			printf("%s : dvt return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "LDL v0v ------ after dvt", v0v, col, row ) ;
#endif 

		sub ( y0, v0v, v0v, col * row ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "LDL v0v ------ after sub ", v0v, col, row ) ;
#endif 


		// x0 = (v0h + v0v)./2;

		add ( v0h, v0v, x0, col * row ) ;

		div_h ( x0, 2, col * row ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "x0 ------ after div_h ", x0, col, row ) ;
#endif 

		if ( i == ( iter - 1 ))
			break ;

		// zh = clip(zh + 1/alpha*dh(x0), lambda/2);

		if ( !dh ( x0, v0h, col, row, 0 ))
		{
			printf("%s : dh return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0h ------ after dh ", v0h, col - 1, row ) ;
#endif 

		mul ( v0h, 1/alpha, ( col - 1 ) * row ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0h ------ after mul ", v0h, col - 1, row ) ;
#endif 

		add ( v0h, zh, v0h, ( col - 1 ) * row ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0h ------ after add ", v0h, col - 1, row ) ;
#endif 

		clip ( v0h, zh, col * row, lambda / 2 ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "zh ------ after clip", zh, col - 1, row ) ;
#endif 

		// zv = clip(zv + 1/alpha*dv(x0), lambda/2);

#ifdef CUDA_OBS 
		p_num_nm_f ( "LDL ------ before h_dv x0", x0, col, row ) ;
		p_num_nm_f ( "LDL ------ before h_dv v0v", v0v, col, row ) ;

#endif 

		if ( !dv ( x0, v0v, col, row, 0 ))
		{
			printf("%s : dv return err i %d \n", __func__, i ) ;
			return ( 0 ) ;
		}

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0v ------ after dv ", v0v, col, row - 1 ) ;
#endif 

		mul ( v0v, 1/alpha, col * ( row - 1 )) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0v ------ after mul ", v0v, col, row - 1 ) ;
#endif 

		add ( v0v, zv, v0v, col * ( row - 1 )) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "v0v ------ after add ", v0v, col, row - 1 ) ;
#endif 

		clip ( v0v, zv, col * row, lambda / 2 ) ;

#ifdef CUDA_OBS 
		p_num_nm_f ( "zv ------ after clip", zv, col, row -1 ) ;
#endif 
	}

	buf_put(( char *) zh ) ;
	buf_put(( char *) zv ) ;
	buf_put(( char *) v0h ) ;
	buf_put(( char *) v0v ) ;

	return ( 1 ) ;
}
