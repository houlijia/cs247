#include <stdio.h>
#include "cs_motion_report.h"

void
ma_report_header ( FILE *ofd, int y, int x, int t, int vr, int hr, int tr )
{
	fprintf( ofd, "****==== video info (1) ====****\n") ;
	fprintf( ofd, "vid_size_v,vid_size_h,vid_size_t,uv_ratio_v,uv_ratio_h,uv_ratio_t\n") ;
	fprintf( ofd, "I,I,I,I,I,I\n") ;

	fprintf( ofd, "%d,%d,%d,%d,%d,%d\n", y, x, t, vr, hr, tr ) ;

	fprintf( ofd, "****==== encoder measurements analysis ====****\n") ;
	
	fprintf( ofd, "indx_v,indx_h,indx_t,"
		"ofst_v,ofst_h,ofst_t,"
		"len_v,len_h,len_t,"
		"ovlp_b_v,ovlp_b_h,ovlp_b_t,"
		"ovlp_f_v,ovlp_f_h,ovlp_f_t,"
		"w_blk,w_type,"
		"mxv,mxp_v,mxp_h,mxp_t,"
		"mdv,mdp_v,mdp_h,mdp_t,"
		"vlc_v,vlc_h\n") ;

	fprintf( ofd, "I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,F,I,I,I,F,I,I,I,F,F\n") ;
}

void
ma_report_record ( FILE *ofd, int *dp, int cnt, int xs, int ys, int zs,
	int blk_in_x, int blk_in_y, int ovh, int ovv, int ovt, int wblk, int wtype )
{
	int	t, v, h, va, ot, ov, oh, ova ;
	int xss, yss, j, k ;
	int off_h, off_v, off_t ;
	int fovh, fovv, fovt, bovh, bovv, bovt ;

	off_t = cnt * ( zs - ovt ) ;

	if ( cnt == 1 )
	{
		bovt = 0 ;
		fovt = ovt ;
	} else
	{
		bovt = ovt ;
		fovt = ovt ;
	}

	for ( j = 1 ; j <= blk_in_y ; j++ )
	{
		if ( j == 1 )
			off_v = 0 ;
		else
			off_v = ( j - 2 ) * ovv ;

		if (( j == 1 ) || ( j == blk_in_y ))
			yss = ys >> 1 ;
		else
			yss = ys ;

		if ( j == 1 )
		{
			fovv = ovv ;
			bovv = 0 ;
		} else if ( j == blk_in_y )
		{
			fovv = 0 ;
			bovv = ovv ;
		} else
		{
			fovv = ovv ;
			bovv = ovv ;
		}

		for ( k = 1 ; k <= blk_in_x ; k++ )
		{ 
			t = *dp++ ;
			v = *dp++ ;
			h = *dp++ ;
			va = *dp++ ;

			ot = *dp++ ;
			ov = *dp++ ;
			oh = *dp++ ;
			ova = *dp++ ;

			if ( k == 1 )
				off_h = 0 ;
			else
				off_h = ( k - 2 ) * ovh ;

			if (( k == 1 ) || ( k == blk_in_x ))
				xss = xs >> 1 ;
			else
				xss = xs ;
			
			if ( k == 1 )
			{
				fovh = ovh ;
				bovh = 0 ;
			} else if ( k == blk_in_x )
			{
				fovh = 0 ;
				bovh = ovh ;
			} else
			{
				fovh = ovh ;
				bovh = ovh ;
			}

			fprintf( ofd, "%d,%d,%d,%d,%d,%d,"
				"%d,%d,%d,%d,%d,%d,%d,%d,%d,"
				"%d,%d,"
				"%f,%d,%d,%d,%f,%d,%d,%d,"
				"%f,%f\n",
				j,k,cnt,	// index
				off_v, off_h, off_t,
				yss, xss, zs,
				bovv, bovh, bovt,
				fovv, fovh, fovt,
				wblk, wtype,
				(float)va / 1000, v, h, t,
				(float)ova / 1000, ov, oh, ot,
				(( float )ov -( float )v ) / (float)t,
				(( float )oh -( float )h ) / (float)t ) ;
		}
	}
}
