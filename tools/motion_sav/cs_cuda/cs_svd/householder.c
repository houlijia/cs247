#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "householder.h"

static double *t_data ;

static double *c_vec, *r_vec ; 
static int c_vec_cnt, r_vec_cnt ;
static column_only = 0 ;

void svd_getdata( char *fn, int r, int c ) ;

void p_double_mn_matlab( char *s, double *dp, int m, int n ) ;

void
usage( char *s )
{
	fprintf( stderr, "Usage %s -d row column -i datafilename -C \n", s) ;
	fprintf( stderr, "	-d : matrix dimension\n") ;
	fprintf( stderr, "	-C : Column only \n") ;
	fprintf( stderr, "	-i : input file name\n") ;
}

main( int ac, char *av[] )
{
	int reduce_cnt, fd, column = -1, row = -1, m, n, i, j, k ;
	char opt ;
	double *cp, *c_vecp, *r_vecp ;
	char *finname ;
	int opt_num[5] ;

	setbuf( stderr, NULL ) ;
	setbuf( stdout, NULL ) ;

	while ((opt = getopt(ac, av, "Ci:d")) != -1)
	{
		switch (opt) {
		case 'C' :
			column_only = 1 ;
			break ;

		case 'i' :
			finname = optarg ;
			break ;

		case 'd' :
			if ( !get_nums( ac, av, optind, 2, opt_num ))
			{
				exit( 1 ) ;
			}
			row = opt_num[0] ;
			column = opt_num[1] ;

			break ;
		}
	}

	if ( column <= 0 || row <= 0 || finname == NULL )
	{
		usage( av[0] ) ;
		exit( 2 ) ;
	}

	fprintf( stderr, "file %s d %d %d column_only %d \n",
		finname, row, column, column_only ) ;

	c_vecp = c_vec = malloc ( sizeof ( double ) * row * column ) ;
	r_vecp = r_vec = malloc ( sizeof ( double ) * row * column ) ;

	fprintf( stderr, "c_vec %p r_vec %p \n", c_vec, r_vec ) ;

#ifdef CUDA_DBG 

	i = row * column ;
	while ( i-- )
		*c_vecp++ = *r_vecp++ = 333.3333 ;

	c_vecp = c_vec ;
	r_vecp = r_vec ;
#endif 

	c_vec_cnt = r_vec_cnt = 0 ;

	if ( !c_vecp || !r_vecp )
	{
		fprintf( stderr, "error malloc vec\n") ;
		exit( 2 ) ;
	}

	svd_getdata( finname, row, column ) ;

	cp = t_data ;
	n = column ;
    m = row ;

	reduce_cnt = ( row > column ) ? column : row ; // min
	for ( i = 0 ; i < reduce_cnt ; i++ )
	{
		j = i ;
		while ( j-- )
			*c_vecp++ = 0.0 ;

		if ( hh_matrix_reduction_C( cp, column, n, m, c_vecp ))
		{
			c_vec_cnt += row ;
			c_vecp += m ;
		}

		if ( !column_only )
		{
			if ( n > 2 )
			{
#ifdef CUDA_DBG 
				fprintf( stderr, "do row: i %d r_vecp %p\n", i, r_vecp ) ;
#endif 
				j = i + 1 ;
				while ( j-- )
					*r_vecp++ = 0.0 ;

				if ( hh_matrix_reduction_R( cp + 1, column, n - 1, m, r_vecp ))
				{
					r_vec_cnt += column ;
					r_vecp += ( n - 1 ) ;
				}
			}
		}

#ifdef CUDA_DBG 
#ifdef CUDA_OBS 
		p_double("column vector", c_vec, c_vec_cnt ) ;
		if ( !column_only )
			p_double("row vector", r_vec, r_vec_cnt ) ;
#endif 
		p_double_mn_matlab("after one iteration V =============", c_vec, row, c_vec_cnt/row ) ;
		if ( !column_only )
			p_double_mn_matlab("after one iteration R =============",
				r_vec, column, r_vec_cnt/column ) ;
		p_double_mn("after one iteration =============", t_data, row, column ) ;
		fprintf( stderr, "m %d n %d ---------------------------------------------\n",
				m, n ) ;
#endif 
		m-- ;
		n-- ;

		cp += ( column + 1 ) ;
	}
}

void
svd_getdata( char *fn, int r, int c )
{
	FILE *fp ;
	int i, j ;
	double *dp ;

	fp = fopen ( fn, "r" ); 

	if ( fp == NULL )
	{
		fprintf( stderr, "file open error %s\n", fn ) ;
		exit( 1 ) ;
	}

	t_data = malloc ( sizeof ( double ) * r * c ) ;

	if ( t_data == NULL ) 
	{
		fprintf( stderr, "malloc failed \n" ) ;
		exit( 1 ) ;
	}

	i = r * c ;
	dp = t_data ;
	while ( i-- )
	{
		if ( fscanf( fp, "%d", &j ) == EOF )
		{
			fprintf( stderr, "scan failed r %d c %d i %d total %d\n",
				r, c, i, r*c ) ;
			exit( 1 ) ;
		}
		*dp++ = ( double ) j ;
	}

	// p_double( "input ---", t_data, r*c ) ;
	p_double_mn ("input --- ", t_data, r, c ) ; 
}

/*
hh_get_norm: good for Column and Row
cp:
	the first entry of the sub-matrix
Row:
	column_data_cnt: 1
	row_cnt: number of n in the sub-matrix
Column:
	column_data_cnt: number of n in the original matrix
	row_cnt: num of m in the sub-matrix
*/
int
hh_get_norm( double *cp, int column_data_cnt, int row_cnt,
	double *normp )
{
	int i ;
	double d, ds ;

	d = 0.0 ;
	while ( row_cnt-- )
	{
#ifdef CUDA_OBS 
		fprintf( stderr, "%s: a %f\n", __func__, *cp ) ;
#endif 
		d += ( *cp ) * ( *cp ) ;
		cp += column_data_cnt ;
	}

	ds = sqrt( d ) ;

	*normp = ds ;
	return ( 1 ) ;
}

/*
hh_d11: good for Column and Row (eq-16 eq-23)
cp:
	the first entry of the sub-matrix
Row:
	column_data_cnt: 1
	row_cnt: number of n in the sub-matrix
Column:
	column_data_cnt: number of n in the original matrix
	row_cnt: num of m in the sub-matrix
*/
int
hh_d11( double *cp, int column_data_cnt, int row_cnt, double *d11p )
{
	double norm ;
	int sign = *cp ;

	hh_get_norm( cp, column_data_cnt, row_cnt, &norm ) ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: norm %f\n", __func__, norm ) ;
#endif 

	if ( sign >= 0 )
		*d11p = -norm ;
	else
		*d11p = norm ;	

	return ( 1 ) ;
}

/*
cp:	"first" matrix data starts from here
column_data_cnt: number of column in the "first" matrix
column_cnt: number of column in the current sub-matrix
row_cnt: number of row in the current sub-matrix
*/

/*
hh_matrix_reduction_R:
	cp: the first entry of the sub-matrix
	column_data_cnt: number of m in the original matrix
	row_cnt: number of m in the sub-matrix
	column_cnt: number of n in the sub-matrix 
*/
int
hh_matrix_reduction_R( double *cp, int column_data_cnt, int column_cnt,
	int row_cnt, double *rvecp )
{
	double f1, d11, w11, a11 ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: cp %p column_offset %d c %d r %d rvecp %p\n",
		__func__, cp, column_data_cnt, column_cnt, row_cnt, rvecp ) ;
#endif 

	if ( column_cnt <= 1 )
		return ( 0 ) ;

	a11 = ( double )*cp ;
	hh_d11( cp, 1, column_cnt, &d11 ) ;

	w11 = a11 - d11 ;

#ifdef CUDA_DBG 
	fprintf( stderr, "a11 %f d11 %f w11 %f\n", a11, d11, w11 ) ;
#endif 

	f1 = sqrt( -2.0 * w11 * d11 ) ;

#ifdef CUDA_DBG 
	fprintf( stderr, "f1 %f\n", f1 ) ;
#endif 

	hh_make_v( rvecp, cp, 1, column_cnt, w11, f1 ) ; 

	hh_apply_fi_R ( rvecp, cp, column_data_cnt, column_cnt, row_cnt ) ;

	return ( 1 ) ;
}

/*
hh_matrix_reduction_C:	eq-24
cp: the first entry of the sub-matrix
column_data_cnt: number of n in the original matrix
row_cnt: num of m in the sub-matrix
column_cnt: number of n in the sub-matrix
cvecp: left vector matrix
*/
int
hh_matrix_reduction_C( double *cp, int column_data_cnt, int column_cnt,
	int row_cnt, double *cvecp )
{
	double f1, d11, w11, a11 ;

#ifdef CUDA_DBG 
	fprintf( stderr, "%s: cp %p column_offset %d c %d r %d\n",
		__func__, cp, column_data_cnt, column_cnt, row_cnt ) ;
#endif 

	if ( row_cnt <= 1 )
		return ( 0 ) ;

	a11 = ( double )*cp ;
	hh_d11( cp, column_data_cnt, row_cnt, &d11 ) ;

	w11 = a11 - d11 ;

#ifdef CUDA_DBG 
	fprintf( stderr, "a11 %f d11 %f w11 %f\n", a11, d11, w11 ) ;
#endif 

	f1 = sqrt( -2.0 * w11 * d11 ) ;

#ifdef CUDA_DBG 
	fprintf( stderr, "f1 %f\n", f1 ) ;
#endif 

	hh_make_v( cvecp, cp, column_data_cnt, row_cnt, w11, f1 ) ; 

	hh_apply_fi_C ( cvecp, cp, column_data_cnt, column_cnt, row_cnt ) ;

	return ( 1 ) ;
}

/*
hh_apply_f1: good for Column and Row
ovp :
	vector table
ocp :
	first entry in the row/column
offset:
	Row: 1
	Column: m in the original matrix
cnt:
	Row: column count in the sub-matrix
	Column: row count in the sub-matrix
*/
void
hh_apply_fi( double *ovp, double *ocp, int offset, int cnt )
{
	double *cp, *vp, f_i ;
	int i, j ;

	// calculate the f_i
	f_i = 0.0 ;
	i = cnt ;
	cp = ocp ;
	vp = ovp ;
	while ( i-- )
	{
		f_i += *vp * *cp ;

#ifdef CUDA_OBS 
		fprintf( stderr, "%s: %d f_i %f vp %f cp %f\n",
			__func__, i, f_i, *vp, *cp ) ;
#endif 
		vp++ ;
		cp += offset ;	   
	}

	f_i *= 2 ; 

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: final f_i %f\n",__func__, f_i ) ;
#endif 

	// apply the f_i to make Ha_i

	vp = ovp ;
	cp = ocp ;
	i = cnt ;
	while ( i-- )
	{
		*cp = *cp - ( f_i * ( *vp )), 
#ifdef CUDA_OBS 
		fprintf( stderr, "%s: i %d f_i %f vp %f cp %f\n",
			__func__, i, f_i, *vp, *cp ) ;
#endif 
		cp += offset ;	   
		vp++ ;
	}
}

/*
eq(24) of the paper

hh_apply_f1_R:
	ovp: the first entry of the vector
	ocp: the first entry of the sub-matrix
	column_data_cnt: number of m in the original matrix
	row_cnt: number of n in the sub-matrix
	column_cnt: number of m in the sub-matrix
*/
void
hh_apply_fi_R( double *ovp, double *ocp, int column_data_cnt, int column_cnt,
	int row_cnt )
{
	double *cp ;
	int k ;

	for ( k = 0 ; k < row_cnt ; k++ )	// size is n in mxn matrix
	{
		cp = ocp + ( k * column_data_cnt ) ;
		hh_apply_fi( ovp, cp, 1, column_cnt ) ;
	}
}

/*
eq(24) of the paper

hh_apply_f1_C: 
	ovp: the first entry of the vector
	ocp: the first entry of the sub-matrix
	column_data_cnt: number of m in the original matrix
	row_cnt: num of m in the sub-matrix
	column_cnt: num of n in the sub-matrix
*/
void
hh_apply_fi_C( double *ovp, double *ocp, int column_data_cnt, int column_cnt,
	int row_cnt )
{
	double *cp ;
	int k ;

	for ( k = 0 ; k < column_cnt ; k++ )	// size is n in mxn matrix
	{
		cp = ocp + k ;
		hh_apply_fi( ovp, cp, column_data_cnt, row_cnt ) ;
	}
}

/*
hh_make_v: good for Column and Row.  part of eq-24
vp:
	space allocated for the vector
	Row: number of n in the sub-matrix
	Column: number of m in the sub-matrix
cp:
	the first entry of the sub-matrix
Row:
	column_data_cnt: 1
	row_cnt: number of n in the sub-matrix
Column:
	column_data_cnt: number of m in the original matrix
	row_cnt: num of m in the sub-matrix
*/
int
hh_make_v( double *vp, double *cp, int column_data_cnt,
	int row_cnt, double w11, double f1 )
{
	double *ovp = vp ;

#ifdef CUDA_DBG 
	int old_row_cnt = row_cnt ;
#endif 
	if ( f1 == 0.0 )
		return ( 0 ) ;

	*vp++ = w11/f1 ;
	cp += column_data_cnt ;
	row_cnt-- ;
	while ( row_cnt-- )
	{
		*vp = *cp / f1 ; 
#ifdef CUDA_OBS 
		fprintf( stderr, "%s: v %f a %f f1 %f\n", __func__, *vp, *cp, f1 ) ;
#endif 
		vp++ ;
		cp += column_data_cnt ;
	}

#ifdef CUDA_OBS 
	p_double( "vector", ovp, old_row_cnt ) ;
#endif 

	return ( 1 ) ;
}

//

#ifdef CUDA_OBS 
int
Houspre(  
Housprod( double *dp, int m, int n )
{

}
#endif 

// dbg funcs ...

void
p_double_mn_matlab( char *s, double *dp, int m, int n )
{
	int j, i ;
	double *odp = dp ;

	fprintf( stderr, "%s: %s ... m %d n %d \n", __func__, s, m, n ) ;

	for ( i = 0 ; i < m ; i++ )
	{
		dp = odp + i ;
		fprintf( stderr, "m == %d --\n", i ) ;
		for ( j = 0 ; j < n ; j++ )
		{
	   		fprintf( stderr, "%f ", *dp ) ;
			dp += m ;
		}
		fprintf( stderr, "\n") ;
	}
}	   

void
p_double_mn( char *s, double *dp, int m, int n )
{
	int j, i ;

	fprintf( stderr, "%s: %s ...\n", __func__, s ) ;

	for ( i = 0 ; i < m ; i++ )
	{
		fprintf( stderr, "m == %d --\n", i ) ;
		for ( j = 0 ; j < n ; j++ )
		{
	   		fprintf( stderr, "%f ", *dp++ ) ;
		}
		fprintf( stderr, "\n") ;
	}
}	   

void
p_double( char *s, double *dp, int cnt )
{
	int i ;

	fprintf( stderr, "%s: %s ... dp %p cnt %d\n", __func__, s, dp, cnt ) ;
	for ( i = 0 ; i < cnt ; i++ )
	   fprintf( stderr, "%d %f\n", i, *dp++ ) ;
}	   

// cmd line options ... 

int
alldigit( char *s )
{
	while ( *s )
	{
		if (!( isdigit( *s )))
			return ( 0 ) ;
		s++ ;
	}
	return ( 1 ) ;
}

int
get_nums( int ac, char *av[], int idx, int cnt, int *np )
{
	if (( idx + cnt ) <= ac )
	{
		while ( cnt-- )
		{
			if ( alldigit( av[ idx ] ))
				*np++ = atoi ( av[ idx++ ] ) ;
			else
				return ( 0 ) ;
		}
		return ( 1 ) ;
	} else
	{
		printf("not enough av idx %d cnt %d ac %d\n", idx, cnt, ac ) ;
		return ( 0 ) ;
	}
}
