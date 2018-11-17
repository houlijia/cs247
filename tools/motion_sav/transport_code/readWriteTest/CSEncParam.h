/* 
 * File:   CSEncParam.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 25, 2014, 12:52 PM
 */

#ifndef CSENCPARAM_H
#define	CSENCPARAM_H
#include "CodeElement.h"

/**
 * Created this class just for reading some infomation to calculate the delay.
 * But, it can be further used to get more infomation for other purpose like arithmetic
 * coding. 
 * @param ce
 */
class CSEncParam : public CodeElement
{
public:
 CSEncParam (const CodeElement& ce)
 {
  //first skip 8 int numbers;
    const char* errString = NULL;
  int endP;
  const char* currentP;
  
  currentP = (ce.getData());
 int pos = findPos((const unsigned char*)(currentP), 0, 8, 80, &errString);

  currentP+= pos+1;

  //then read 1_conv_rng
  conv_rng =readOnlyOneUInt(currentP , 0, &endP, &errString); 
  currentP+= endP+1;

  //then skip 2+1_conv_rng int
  pos = findPos((unsigned char*)currentP, 0, conv_rng+2, 10*(conv_rng+2), &errString);
  currentP+= pos+1;
  
  //then read blk_size
  for(int j=0; j<3; j++)
   for(int i=0; i<6; i++)
    {
     blk_info[i][j] =readOnlyOneUInt(currentP , 0, &endP, &errString);
     currentP += endP+1;
    }
  
 }
 //CSEncParam (const CSEncParam& orig);
 virtual ~CSEncParam () {};
 int n_frames;
 int start_frame;
 longInt random_seed;
 uint8 process_color;
 int wnd_type;
 longInt conv_rng;
 int blk_info[6][3];
 //int blk_cnt[2];
 
private:

};

#endif	/* CSENCPARAM_H */

