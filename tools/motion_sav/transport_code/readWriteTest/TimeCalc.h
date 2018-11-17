/* 
 * File:   TimeCalc.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 25, 2014, 2:58 PM
 */

#ifndef TIMECALC_H
#define	TIMECALC_H
#include "RawVideoInfo.h"
#include "CSEncParam.h"
#include <cmath>

class TimeCalc
{
public:
 TimeCalc (RawVideoInfo* ri0, CSEncParam* ep0):ri(ri0), ep(ep0)
 {

 }
/**
 * 
 * @return the time in micro-second 
 */
 int getInterval(int rtpPerBlock)
 {
  int blk_cnt[2];
  blk_cnt[0] = ceil((ri->height - (ep->blk_info[1][0]))/(ep->blk_info[0][0] - ep->blk_info[1][0]));
  blk_cnt[1] = ceil((ri->width- (ep->blk_info[1][1]))/(ep->blk_info[0][1] - ep->blk_info[1][1]));
  float blockTime = ((float)(ep->blk_info[0][2] - ep->blk_info[1][2])) / ((ri->frameRate) * blk_cnt[0] *blk_cnt[1]);
  int re =(blockTime*0.85)/ rtpPerBlock *1000000;
  return re;
  
 } 
 //TimeCalc (const TimeCalc& orig);
 virtual ~TimeCalc (){
  if(ri != NULL)
   {
    delete ri;
    ri = NULL;
   }
  if(ep !=NULL)
   {
    delete ep;
    ep =NULL;
   }
 }
private:
 RawVideoInfo* ri;
 CSEncParam* ep;

};

#endif	/* TIMECALC_H */

