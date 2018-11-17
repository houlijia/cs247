#include <unistd.h>
#include <ctype.h>
#include <stdio.h>
#include <endian.h>
#include <omp.h>
#include <sys/times.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cs_cuda.h"
#include "cs_helper.h"
#include "cs_header.h"
#include "cs_analysis.h"
#include "cs_block.h"

#define CUDA_DBG 
// swap

__global__ void d_expand_c_to_i( char *fp, int *tp, int size )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	while ( tid < size )
	{
		tp[ tid ] = fp [ tid ] ;
		tid += CUDA_MAX_THREADS ;
	} 
}

// clear the device mem ... n in size of int

void
h_expand_c_to_i ( char *din, int *dout, int n )	
{
	int nThreadsPerBlock = 512;
	int nBlocks ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	d_expand_c_to_i <<< nBlocks, nThreadsPerBlock >>> ( din, dout, n ) ;

	cudaThreadSynchronize() ;
}

__global__ void d_swap_an_entry_i( int *a, int size )
{
	int t, tid = blockIdx.x*blockDim.x + threadIdx.x;

	while ( tid < size )
	{
		t = a[ tid ] ;

		a[ tid ] = (( t & 0xff ) << 24 ) |
			(( t & 0xff00 ) << 8 ) |
			(( t & 0xff0000 ) >> 8 ) |
			(( t & 0xff000000 ) >> 24 ) ;

		tid += CUDA_MAX_THREADS ;
	} 
}

// clear the device mem ... n in size of int

void
htonl_device_mem_i ( int *d_a, int n )	
{
#if __BYTE_ORDER__ == __BIG_ENDIAN
	return ;
#else
	int nThreadsPerBlock = 512;
	int nBlocks ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	d_swap_an_entry_i <<< nBlocks, nThreadsPerBlock >>> (d_a, n ) ;

	cudaThreadSynchronize() ;
#endif
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

//  times ... all times are in ticks 

static int ticks_per_second = 1 ;

struct cs_timer {
	clock_t	user_t ;
	clock_t sys_t ;

	clock_t total_user_t ;	
	clock_t total_sys_t ;

	int clock_cnt ;
	int clock_on ;
} ;

static struct cs_timer *cs_timer_tbl ;
static int cs_timer_tbl_size = 0;

int 
cs_timer_init( int cnt )
{
	struct cs_timer *otp ;

	if (( cs_timer_tbl = ( struct cs_timer * )malloc (
		sizeof ( *cs_timer_tbl ) * cnt )) == NULL )
	{
		printf("%s cnt %d malloc failed\n", __func__, cnt ) ;
		return ( 0 ) ;
	} 

	ticks_per_second = sysconf( _SC_CLK_TCK ) ;

	cs_timer_tbl_size = cnt ;

	otp = cs_timer_tbl ;

	while ( cnt-- )
	{
		otp->user_t = 0 ;
		otp->sys_t = 0 ;
		otp->total_user_t = 0 ;
		otp->total_sys_t = 0 ;

		otp->clock_cnt = 0 ;
		otp->clock_on = 0 ;

		otp++ ;
	}

	return ( 1 ) ;
}

int 
cs_timer_on( int idx )
{
	struct cs_timer *otp ;
	struct tms tm ;

	if (!(( idx < cs_timer_tbl_size ) && ( idx >= 0 )))
	{
		fprintf( stderr, "%s: error idx %d size %d\n", __func__, idx, cs_timer_tbl_size ) ;
		return ( 0 ) ;
	}

	otp = cs_timer_tbl + idx ;

	if ( otp->clock_on )
	{
		fprintf( stderr, "%s idx %d timer is already on \n", 
			__func__, idx ) ;
		return ( 0 ) ;
	}

	times( &tm ) ;

	otp->user_t = tm.tms_utime ;
	otp->sys_t = tm.tms_stime ;
	otp->clock_on++ ;

#ifdef CUDA_OBS 
	printf("on timer: u %d s %d tu %d ts %d cnt %d on %d\n",
		otp->user_t,
		otp->sys_t,
		otp->total_user_t,
		otp->total_sys_t,
		otp->clock_cnt,
		otp->clock_on ) ;
#endif 

	return ( 1 ) ;
}

int
cs_timer_off( int idx )
{
	struct cs_timer *otp ;
	struct tms tm ;

	if (!(( idx < cs_timer_tbl_size ) && ( idx >= 0 )))
	{
		fprintf( stderr, "$s: error idx %d size %d\n", idx, cs_timer_tbl_size ) ;
		return ( 0 ) ;
	}

	otp = cs_timer_tbl + idx ;

	if ( !otp->clock_on )
	{
		fprintf( stderr, "%s idx %d timer is already off \n",
			__func__, idx) ;
		return ( 0 ) ;
	}

	times( &tm ) ;

	otp->total_user_t += tm.tms_utime - otp->user_t ;
	otp->total_sys_t += tm.tms_stime - otp->sys_t ;
	otp->clock_on = 0 ;

	otp->clock_cnt++ ;

#ifdef CUDA_OBS 
	printf("off timer: u %d s %d tu %d ts %d cnt %d on %d\n",
		otp->user_t,
		otp->sys_t,
		otp->total_user_t,
		otp->total_sys_t,
		otp->clock_cnt,
		otp->clock_on ) ;
#endif 

	return ( 1 ) ;
}

int
cs_timer_get( int idx, clock_t *sp, clock_t *up, int *cp, double *asp, double *aup )
{
	struct cs_timer *otp ;
	double d ;

	if (!(( idx < cs_timer_tbl_size ) && ( idx >= 0 )))
	{
		fprintf( stderr, "$s: error idx %d size %d\n", idx, cs_timer_tbl_size ) ;
		return ( 0 ) ;
	}

	otp = cs_timer_tbl + idx ;

	*up = otp->total_user_t ;
	*sp = otp->total_sys_t ;

	*cp = otp-> clock_cnt ;

	d = 1000.0 / ticks_per_second ; // ms per tick

#ifdef CUDA_OBS 
	printf("idx %d ut %d st %d d %f cnt %d\n",
		idx,
		otp->total_user_t,
		otp->total_sys_t,
		d,
		otp->clock_cnt ) ;
#endif 

	if ( otp->clock_cnt )
	{
		*aup = (( double )otp->total_user_t * d ) / ( double )otp->clock_cnt ;
		*asp = (( double )otp->total_sys_t * d ) / ( double )otp->clock_cnt ;
	} else
		*aup = *asp = 0.0 ;

	return ( 1 ) ;
}

// omp timing ...

struct omp_timer {
	double omptime ;
	double omptime_start ;
	int ompcnt ;
	int timeron ;
} ;

static struct omp_timer *omp_timer_tbl ;
static int omp_timer_tbl_size = 0;

int 
omp_timer_init( int cnt )
{
	struct omp_timer *otp ;

	if (( omp_timer_tbl = ( struct omp_timer * )malloc (
		sizeof ( *omp_timer_tbl ) * cnt )) == NULL )
	{
		printf("%s cnt %d malloc failed\n", __func__, cnt ) ;
		return ( 0 ) ;
	} 

	omp_timer_tbl_size = cnt ;

	otp = omp_timer_tbl ;

	while ( cnt-- )
	{
		otp->omptime = 0.0 ;
		otp->ompcnt = 0 ;
		otp->timeron = 0 ;
		otp->omptime_start = 0 ;

		otp++ ;
	}

	return ( 1 ) ;
}

int 
omp_timer_on( int idx )
{
	struct omp_timer *otp ;

	if (!(( idx < omp_timer_tbl_size ) && ( idx >= 0 )))
	{
		fprintf( stderr, "$s: error idx %d size %d\n", idx, omp_timer_tbl_size ) ;
		return ( 0 ) ;
	}

	otp = omp_timer_tbl + idx ;

	if ( otp->timeron )
	{
		fprintf( stderr, "%s idx %d timer is already on \n", 
			__func__, idx ) ;
		return ( 0 ) ;
	}
	otp->omptime_start = omp_get_wtime() ;
	otp->timeron++ ;

	return ( 1 ) ;
}

int
omp_timer_off( int idx )
{
	double d ;
	struct omp_timer *otp ;

	if (!(( idx < omp_timer_tbl_size ) && ( idx >= 0 )))
	{
		fprintf( stderr, "$s: error idx %d size %d\n", idx, omp_timer_tbl_size ) ;
		return ( 0 ) ;
	}

	otp = omp_timer_tbl + idx ;

	if ( !otp->timeron )
	{
		fprintf( stderr, "%s idx %d timer is already off \n",
			__func__, idx) ;
		return ( 0 ) ;
	}

	d = omp_get_wtime() ;

	d -= otp->omptime_start ;

	otp->omptime += d ;

	otp->ompcnt++ ;

	otp->timeron = 0 ;

	return ( 1 ) ;
}

int
omp_timer_get( int idx, double *dp, int *cp, double *ddp )
{
	struct omp_timer *otp ;

	if (!(( idx < omp_timer_tbl_size ) && ( idx >= 0 )))
	{
		fprintf( stderr, "$s: error idx %d size %d\n", idx, omp_timer_tbl_size ) ;
		return ( 0 ) ;
	}

	otp = omp_timer_tbl + idx ;

	*dp = otp->omptime * 1000 ; // ms
	*cp = otp->ompcnt ;

	if ( otp->ompcnt )
		*ddp = ( otp->omptime * 1000 ) / ( double )otp->ompcnt ;
	else 
		*ddp = 0.0 ;

	return ( 1 ) ;
}

// misc

// whm will increase that many bits indiated by 'i'
// pixel itself has 8 bits
int
weight_sft( int weight_scheme, int size, int m, int n ) 
{
	int i, j, k ;
#ifdef CUDA_DBG 
	fprintf( stderr, "%s: size %d m %d n %d \n", __func__, size, m, n ) ;
#endif 

	i = max_log2( size ) ;
	i = ( int )log2(( double )i ) ;

#ifdef CUDA_OBS 
	fprintf(stderr, "%s: size %d i %d\n", __func__, size, i ) ;
#endif 

	switch ( weight_scheme ) {
	case WEIGHT_LINEAR :

		j = ( m >> 1 ) ;
		if ( m & 1 )
			j++ ;

		k = ( n >> 1 ) ;
		if ( n & 1 )
			k++ ;
		j *= k ;

		k = max_log2 ( j ) ;
		k = ( int )log2(( double )k ) ;

#ifdef CUDA_OBS 
		fprintf( stderr, "%s: j %d k %d m %d n %d \n", __func__, j, k, m, n ) ;

#endif 
		i += k ;
		break ;

	case NO_WEIGHT :
	default:
		;
	}

	i += 8 ; // pixel bit size

	i -= 31 ;

	if ( i < 0 )
		i = 0 ;

	fprintf( stderr, "%s: shift %d\n", __func__, i ) ;

	return ( i ) ;
}
	
int
max_log2( int i )
{
	int k, j ;

	k = ( int )log2(( double )i ) ;
	j = (int)pow(2.0, k ) ;

	if ( j < i )
		j = (int)pow(2.0, k + 1 ) ;

	return ( j ) ;
}

//  TSTING TSTING ... start

__global__ void d_tst_longlong( long long *a, int size )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid < size )
	{
		a[ tid ] = tid ;
	} 
}

// clear the device mem ... n in size of int
void
h_tst_longlong ( long long *d_a, int n )	
{
	int nThreadsPerBlock = 512;
	// int nBlocks= ( n + 1 )/nThreadsPerBlock ;
	// int nBlocks= n/nThreadsPerBlock + ((n%nThreadsPerBlock)?1:0);
	int nBlocks= ( n + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;

	fprintf( stderr, "h_tst_longlong %p size %d\n", d_a, n ) ;

	d_tst_longlong <<< nBlocks, nThreadsPerBlock >>> (d_a, n) ;

	cudaThreadSynchronize() ;
}

//  TSTING TSTING ... end

// clear memory on device, skip the first "skip" elements

__global__ void d_set_abs( int *a, int size, int len, int skip )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	while ( tid < size )
	{
		a += ( tid / len ) * ( len + skip ) + skip + tid % len ;

		if ( *a < 0 )
			*a = -(*a) ;

		tid += CUDA_MAX_THREADS ;
	} 
}

// record_len: not includes the skip cnt ...
// n: total number of element ... should be multi of record_len
int
h_set_abs ( int *d_a, int n, int record_len, int skip )	
{
	int nThreadsPerBlock = 512;
	int nBlocks ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	if ( n % record_len )
	{
		fprintf( stderr, "%s: error: n %d record len %d\n",
			__func__, n, record_len ) ;
		return ( 0 ) ;
	}

	d_set_abs <<< nBlocks, nThreadsPerBlock >>> (d_a, n, record_len, skip ) ;

	cudaThreadSynchronize() ;

	return ( 1 ) ;
}

// clear memory on device

__global__ void d_set_an_entry_i( int *a, int size, int val )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	while  ( tid < size )
	{
		a[ tid ] = val ;

		tid += CUDA_MAX_THREADS ;
	} 
}

// clear the device mem ... n in size of int
void
set_device_mem_i ( int *d_a, int n, int val )	
{
	int nThreadsPerBlock = 512;
	int nBlocks ;

#ifdef CUDA_OBS 
	fprintf( stderr, "%s: dp %p cnt %d val %d\n", __func__, d_a, n, val ) ;
#endif 

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;
	d_set_an_entry_i <<< nBlocks, nThreadsPerBlock >>> (d_a, n, val ) ;

	cudaThreadSynchronize() ;
}

// clear the device mem ... n in size of int
void
clear_device_mem_i ( int *d_a, int n )	
{
	set_device_mem_i ( d_a, n, 0 ) ;
}
 
// clear memory on device

__global__ void d_set_an_entry_c( char *a, int size, char val )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	while ( tid < size )
	{
		a[ tid ] = val ;

		tid += CUDA_MAX_THREADS ;
	} 
}

// clear the device mem ... n in size of char
void
set_device_mem_c ( char *d_a, int n, char c )	
{
	int nThreadsPerBlock = 512;
	int nBlocks ;

	h_block_adj ( n, nThreadsPerBlock, &nBlocks ) ;

	d_set_an_entry_c <<< nBlocks, nThreadsPerBlock >>> (d_a, n, c ) ;

	cudaThreadSynchronize() ;
}
 
// clear the device mem ... n in size of char
void
clear_device_mem_c ( char *d_a, int n )	
{
	set_device_mem_c ( d_a, n, 0 ) ;
}
 
// To convert a-f or A-F to a decimal number
int 
chartoint(int c)
{
	char hex[] = "aAbBcCdDeEfF";
	int i;
	int result = 0;

	for(i = 0; result == 0 && hex[i] != '\0'; i++)
	{
		if(hex[i] == c)
		{
			result = 10 + (i / 2);
		}
	}

	return result;
}

unsigned int
htoi(const char s[])
{
	unsigned int result = 0;
	int i = 0;
	int proper = 1;
	int temp;

	//To take care of 0x and 0X added before the hex no.
	if(s[i] == '0')
	{
		++i;
		if(s[i] == 'x' || s[i] == 'X')
		{
			++i;
		}
	}

	while(proper && s[i] != '\0')
	{
		result = result * 16;
		if(s[i] >= '0' && s[i] <= '9')
		{
			result = result + (s[i] - '0');
		}
		else
		{
			temp = chartoint(s[i]);
			if(temp == 0)
			{
				proper = 0;
			}
			else
			{
				result = result + temp;
			}
		}

		++i;
	}
	//If any character is not a proper hex no. ,  return 0
	if(!proper)
	{
		result = 0;
	}

	return result;
}

// set the config info

__global__ void d_set_config( struct cs_xyz *a,
	int x0, int y0, int z0,
	int x1, int y1, int z1,
	int x2, int y2, int z2 )
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid < 1 )
	{
		a[0].x = x0 ;
		a[0].y = y0 ;
		a[0].z = z0 ;
		a[1].x = x1 ;
		a[1].y = y1 ;
		a[1].z = z1 ;
		a[2].x = x2 ;
		a[2].y = y2 ;
		a[2].z = z2 ;
	} 
}

// clear the device mem ... n in size of int

void
h_set_config( struct cs_xyz *d_a, struct cube *hp )	
{
#ifdef CUDA_OBS 
	fprintf( stderr, "%s: 0:%d %d %d 1: %d %d %d 2: %d %d %d\n",
		__func__,
		hp[0].x, hp[0].y, hp[0].z,
		hp[1].x, hp[1].y, hp[1].z,
		hp[2].x, hp[2].y, hp[2].z ) ;
#endif 
	d_set_config <<< 1, 1 >>> ( d_a,
		hp[0].x, hp[0].y, hp[0].z,
		hp[1].x, hp[1].y, hp[1].z,
		hp[2].x, hp[2].y, hp[2].z ) ;

	cudaThreadSynchronize() ;
}

void
h_block_adj ( int n, int nThreadsPerBlock, int *block )
{
	if ( n > CUDA_MAX_THREADS )
		*block = CUDA_MAX_BLKS ;
	else
		*block = ( n + ( nThreadsPerBlock - 1 ))/nThreadsPerBlock ;
}

int
put_d_data_i ( int *dp, int *hp, int size ) 
{
	int i ;

	if (( i = cudaMemcpy( dp, hp, size, cudaMemcpyHostToDevice)) !=
		cudaSuccess )
	{
		fprintf( stderr, "%s: failed %d\n", __func__, i ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}

int
get_d_data_i ( int *dp, int *hp, int size ) 
{
	int i ;

	if (( i = cudaMemcpy( hp, dp, size, cudaMemcpyDeviceToHost)) !=
		cudaSuccess )
	{
		fprintf(stderr, "%s: failed %d\n", __func__, i ) ;
		return ( 0 ) ;
	}
	return ( 1 ) ;
}
