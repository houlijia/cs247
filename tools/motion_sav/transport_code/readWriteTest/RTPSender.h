/* 
 * File:   RTPSender.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 21, 2014, 11:55 AM
 */

#ifndef RTPSENDER_H
#define	RTPSENDER_H
#include "RtpPacket.h"

class RTPSender
{
public:
 RTPSender () {};
 //RTPSender (const RTPSender& orig);
 virtual int send(const RtpPacket* const rtpP, const char** errString) =0;
 virtual ~RTPSender (){};
private:

};

#endif	/* RTPSENDER_H */

