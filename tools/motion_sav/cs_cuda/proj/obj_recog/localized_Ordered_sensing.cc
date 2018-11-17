#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../../cs_misc/cs_helper.h"

#include "localized_Ordered_sensing.h"

#define TARGET_CELLS	6

extern int max_log2( int i )  ;

#ifdef CUDA_OBS 
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
#endif 

// the input is 1 colors ... we only use 1
// ip points to orig_w * orig_h of one color
int *
lo_sensing( int *ip, int orig_w, int orig_h, int w, int h, int w_offset,
	int h_offset ) 
{
	int bl1, bl2, bl3, bl4, j, i, t_w, t_h ;
	int *tp ;
	int t_temp, *fp1, *fp2, *fp3, *fp4, *tp1, *tp2, *tp3, *tp4, *tp5, *tp6 ;

	i = max_log2( h ) ;

	printf("%s: w %d h %d log %d w %d %d %d g %d %d %d\n",
		__func__, w, h, i, orig_w, w, w_offset, orig_h, h, h_offset ) ;

	if (( w != h ) || (( orig_w - w_offset ) < w ) ||
		(( orig_h - h_offset ) < h ) || ( i != h ))
	{
		printf("%s: err w %d h %d log %d w %d %d %d g %d %d %d\n",
			__func__, w, h, i, orig_w, w, w_offset, orig_h, h, h_offset ) ;
		return ( NULL ) ;
	}

	t_w = t_h = w / 4 ;

	fp1 = ip + orig_w * h_offset + w_offset ;
	fp2 = fp1 + orig_w ;
	fp3 = fp2 + orig_w ;
	fp4 = fp3 + orig_w ;

	i = t_w * t_h ;

	tp = ( int * ) malloc ( sizeof ( int ) * t_w * t_h * TARGET_CELLS ) ;

	tp1 = tp ;
	tp2 = tp1 + i ;
	tp3 = tp2 + i ;
	tp4 = tp3 + i ;
	tp5 = tp4 + i ;
	tp6 = tp5 + i ;

	printf("ok 111 \n") ;

	for ( i = 0 ; i < t_h ; i++ )
	{
		for ( j = 0 ; j < t_w ; j++ )
		{
			// printf("i %d j %d \n", i, j ) ;

			bl1 = *fp1 + *( fp1+1 ) + *fp2 + *( fp2 + 1 ) ;
			bl2 = *fp3 + *( fp3+1 ) + *fp4 + *( fp4 + 1 ) ;

			bl3 = *( fp1 + 2 ) + *( fp1 + 3 ) + *( fp2 + 2 ) + *( fp2 + 3 ) ;
			bl4 = *( fp3 + 2 ) + *( fp3 + 3 ) + *( fp4 + 2 ) + *( fp4 + 3 ) ;

			t_temp = *tp1++ = bl1 + bl2 + bl3 + bl4 ;

			*tp2++ = bl1 - bl2 + bl3 - bl4 ;
			*tp3++ = bl1 + bl2 - bl3 - bl4 ;

			*tp4++ = t_temp - 2 *
				( *fp2 + *( fp2 + 1 ) + *( fp2 + 2 ) + *( fp2 + 3 ) +
				*fp3 + *( fp3 + 1 ) + *( fp3 + 2 ) + *( fp3 + 3 )) ;

			*tp5++ = bl1 - bl2 - bl3 + bl4 ;

			*tp6++ = t_temp - 2 *
				(*( fp1 + 1 ) + *( fp1 + 2 ) + 
				*( fp2 + 1 ) + *( fp2 + 2 ) + 
				*( fp3 + 1 ) + *( fp3 + 2 ) + 
				*( fp4 + 1 ) + *( fp4 + 2 )) ;

			fp1 += 4 ;
			fp2 += 4 ;
			fp3 += 4 ;
			fp4 += 4 ;
		}

		fp1 += 3 * orig_w ;
		fp2 = fp1 + orig_w ;
		fp3 = fp2 + orig_w ;
		fp4 = fp3 + orig_w ;
	}

	return ( tp ) ;
}

