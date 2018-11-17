#ifndef __CS_MOTION_REPORT__
#define __CS_MOTION_REPORT__


void ma_report_header ( FILE *ofd, int y, int x, int t, int vr, int hr, int tr ) ;
void ma_report_record ( FILE *ofd, int *dp, int cnt, int xs, int ys, int zs,
	int blk_in_x, int blk_in_y, int ovh, int ovv, int ovt, int wblk, int wtype )  ;

#endif 
