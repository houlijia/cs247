/* 
 * File:   UdpRTPReceiver.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 21, 2014, 5:16 PM
 */
#include "RTPReceiver.h"
#include "udp_client_server.h"

#ifndef UDPRTPRECEIVER_H
#define	UDPRTPRECEIVER_H
#include <iostream>

static const int MAX_UDP_PACKET_LEN = 2000;

using std::cerr;
using std::endl;

/**
 * This is the class for 
 * @param fromIP0 bind to this ip to receive
 * @param port0 Bind to this port
 * @param timeout0 This timeout is used for both select and recv, in ms
 * The alarm in the cygwin platform will be timout/1000+2;
 */
class UdpRTPReceiver : public RTPReceiver
{
public:
 UdpRTPReceiver (const string& fromIP0, const int port0, const int timeout0):
 fromIP(fromIP0), port(port0), timeout(timeout0)
 {
  revp = new udp_server(fromIP, port); 
  buffer = (char*) malloc(MAX_UDP_PACKET_LEN ); 
 }
 
/**
 * If we get a packet from the network, new a RtpPacket based on the received
 * data, and return the pointer
 * @param errString
 * @return the RtpPacket pointer newed 
 */ 
 RtpPacket* getPacket(const char** errString)
 {
  RtpPacket* rtpP = NULL;
  int recvLen = revp->timed_recv (buffer, MAX_UDP_PACKET_LEN, timeout);
  if(recvLen >12)
   {
    rtpP = new RtpPacket(buffer, recvLen);
   }
  return rtpP;
 }
 virtual ~UdpRTPReceiver (){
  if(revp != NULL)
   {
    delete revp;
    revp = NULL;
   }
  if(buffer != NULL)
   {
    free(buffer);
    buffer = NULL;
   }
  
 }
private:
 udp_server* revp; /** < the pointer to the receiver*/
 const string& fromIP;
 const int port;
 const int timeout;
 char* buffer; /** < The buffer for reading*/
 
};

#endif	/* UDPRTPRECEIVER_H */

