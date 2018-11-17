#include "RtpPacket.h"

int
main ()
{
  char* data = "hello rtp!";
  char* errString = NULL;

  //	RtpPacket(uint32 timeStamp, uint16 seq, uint8 payloadType,  uint32 ssrc, char* dataSrc, uint32 dataLen0):dataLen(dataLen0)
  RtpPacket p (1226, 3223, 98, 0xFFFF, data, 11);
  char* folder = "../../../output/20140606_1719/1.case/rtp";
  p.writeToFile (string (folder), &errString);


}

