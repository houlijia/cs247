/* 
 * File:   RTPReceiver.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 21, 2014, 4:40 PM
 */

#ifndef RTPRECEIVER_H
#define	RTPRECEIVER_H
#include "RtpPacket.h"
#include "DropList.h"

class RTPReceiver
{
public:
 RTPReceiver (){};
// RTPReceiver (const RTPReceiver& orig);

 /**
  * @note will new a RtpPacket inside, user should delete it later
  * @param errString
  * @return 
  */
virtual RtpPacket* getPacket(const char** errString) =0;
 void setDrop(AbstractDropList* dl0)
 {
  if(dlp !=NULL)
   delete dlp;
  dlp = dl0;
 }
 
 virtual ~RTPReceiver () {};

protected:
 AbstractDropList* dlp;
};

#endif	/* RTPRECEIVER_H */

