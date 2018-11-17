/* 
 * File:   TcpBufferSender.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 24, 2014, 1:43 PM
 */

#ifndef TCPBUFFERSENDER_H
#define	TCPBUFFERSENDER_H

#include "BufferSender.h"
#include "tcpconnect.h"

class TcpBufferSender :public BufferSender
{
public:
 TcpBufferSender (const char* nameIP, const char* port, int size): BufferSender(size)
 {
  socketNo = tcpconnect_start_client(nameIP, port); 
 }

 uint32 send( uint32 sendLen)
 {
    int sent = write(socketNo, buffer,  sendLen);
  return sent;
 }
 //TcpBufferSender (const TcpBufferSender& orig);
 virtual ~TcpBufferSender ()
 {
  close(socketNo);
 }
private:
 int socketNo;

};

#endif	/* TCPBUFFERSENDER_H */

