/* 
 * File:   UdpRTPSender.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 21, 2014, 11:58 AM
 */

#ifndef UDPRTPSENDER_H
#define	UDPRTPSENDER_H
#include "RTPSender.h"
#include "udp_client_server.h"


class UdpRTPSender : public RTPSender
{
public:
 UdpRTPSender (const string & desAddress, const int& portNumber): destAddr(desAddress),
   port(portNumber), sender(NULL)
 {
  senderBuffer = (char*) malloc(MAX_RTP_DATA_LEN);
  
 }
 void init()
 {
  sender = new udp_client(destAddr, port);
  
 }
 UdpRTPSender (const UdpRTPSender& orig);

 
 /**
  * @note must call init() first, it will copy the rtp packet
  * it will not change it 
  * @param rtp
  * @return 
  */
 int send(const RtpPacket* const rtp, const char** errString)
 {
  assert(sender != NULL);
  int sent =0;
  memcpy(senderBuffer, rtp->getReadOnlyHeader(), rtp->getHeaderLength());
  memcpy(senderBuffer + rtp->getHeaderLength(),rtp->getReadOnlyData(),  rtp->getDataLength() );

  sent = sender->send(senderBuffer, (rtp->getDataLength()) + (rtp->getHeaderLength()));
  return sent;
  
 }
 virtual ~UdpRTPSender ();
 
private:
 string destAddr;
 int port;
 udp_client* sender;
 char* senderBuffer;
 

};

#endif	/* UDPRTPSENDER_H */

