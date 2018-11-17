/* 
 * File:   RawVideoInfo.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 25, 2014, 12:36 PM
 */

#ifndef RAWVIDEOINFO_H
#define	RAWVIDEOINFO_H
#include "CodeElement.h"
#include <iostream>
using std::cerr;
using std::endl;

class RawVideoInfo : public CodeElement
{
public:
 RawVideoInfo (const CodeElement& ce)
 {
  const char* errString = NULL;
  int endP;
  const char* currentP;
  currentP = (ce.getData ()); 
  uint8 pos = findPos((unsigned char*) currentP, 0, 3, 30, &errString);
  currentP +=pos+1;

   width =readOnlyOneUInt(currentP , 0, &endP, &errString); 
  currentP+= endP+1;

   height=readOnlyOneUInt(currentP , 0, &endP, &errString); 
  currentP+= endP+1; 


  pos = findPos((unsigned char*) currentP, 0, 2, 20, &errString);
  
  currentP +=pos+1;
  frameRate =readOnlyOneUInt(currentP , 0, &endP, &errString); 
  cerr<<"wh rate "<<width<<" "<<height<<" "<<frameRate<<endl;
 }
// RawVideoInfo (const RawVideoInfo& orig);
 virtual ~RawVideoInfo (){};
 int frameRate;
 int width;
 int height;
 
private:

};

#endif	/* RAWVIDEOINFO_H */

