/** 
 * @file:   UdpRTPSender.cpp
 * @author: Jianwei Liu<jianwel@g.clemson.edu>
 * 
 * Created on July 21, 2014, 11:58 AM
 */

#include "UdpRTPSender.h"

//UdpRTPSender::UdpRTPSender () { }

//UdpRTPSender::UdpRTPSender (const UdpRTPSender& orig) { }

UdpRTPSender::~UdpRTPSender () {
 if(this->sender !=NULL)
  {
  delete this->sender;
  this->sender = NULL;
  }
 if(senderBuffer != NULL)
  {
   free(senderBuffer);
   senderBuffer = NULL;
  }
}

