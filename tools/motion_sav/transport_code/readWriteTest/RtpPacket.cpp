#include "RtpPacket.h"
#include <cstdio>

RtpPacket::RtpPacket (uint32 timeStamp, uint16 seq, char payloadType, uint32 ssrc, char* dataSrc, uint32 dataLen0) : dataLen (dataLen0)
{
  memset (fixedHeader, 0, 12);
  setVersion ();
  setPayLoadType (payloadType);
  setSeqNo (seq);
  setSSRC (ssrc);
  setTimeStamp (timeStamp);
  this->data = dataSrc;
  csrc = NULL;
}

RtpPacket::RtpPacket (char payloadType, uint32 ssrc, uint32 maxLen) : dataLen (0)
{
  memset (fixedHeader, 0, 12);
  setVersion ();
  setPayLoadType (payloadType);
  setSSRC (ssrc);
  this->data = (char*) malloc (maxLen);
  csrc = NULL;
}

RtpPacket::RtpPacket (FILE* fp)
{
  const char* errString = NULL;
  csrc = NULL;
  data = NULL;
  if(readBasicHeader (fp, &errString))
    {
      //the second parameter is not used now
      readData (fp, MAX_RTP_DATA_LEN, &errString);
    }
  checkError(errString);
}


  RtpPacket::RtpPacket (char* buffer, uint32 len)
  {
  const char* errString = NULL;
  csrc = NULL;
  data = NULL;
 readBasicHeaderFromBuffer (buffer, &errString);
      //the second parameter is not used now
 readDataFromBuffer(buffer + this->getHeaderLength(), len- (this->getHeaderLength()), &errString);
  checkError(errString);

   
  }

char
RtpPacket::readBasicHeader (FILE* fp, const char** errString)
{
  char readed = fread (fixedHeader, 1, this->getHeaderLength(), fp);
  if(readed != this->getHeaderLength())
    {
      *errString = "read basic header not 12 in RtpPacket\n";
      fputs (*errString, stderr);
      return 0;
    }
  else
    return 1;
}

char
RtpPacket::readBasicHeaderFromBuffer (char* buffer, const char** errString)
{
  memcpy(fixedHeader, buffer,  this->getHeaderLength());
  return 1;
  
}

uint32
RtpPacket::readData (FILE* fp, uint32 toReadLen, const char** errString)
{
  if(data == NULL)
    data = (char*) malloc (MAX_RTP_DATA_LEN);
  this->dataLen = fread (data, 1, MAX_RTP_DATA_LEN, fp);
  return this->dataLen;

}

uint32
RtpPacket::readDataFromBuffer (char* buffer, uint32 toReadLen, const char** errString)
{
  if(data == NULL)
    data = (char*) malloc (toReadLen);
   memcpy(data, buffer, toReadLen);
   this->dataLen = toReadLen;
  return this->dataLen;

}


uint16
RtpPacket::writeToFile (string folderName, uint16 baseNumber, const char**errString) const
{
  uint16 fileLen = 0;
  uint16 written = 0;
  string finalFileName = folderName + "/" + intToString(this->getSeqNo () - baseNumber, 5) + ".rtp";
  FILE* fp = fopen (finalFileName.c_str (), "wb");
  if(fp == NULL)
    {
      *errString = "open rtp file error in writeToFile";
      return 0;
    }
  written = fwrite (fixedHeader, 1,this->getHeaderLength() , fp);
  fileLen += written;
  if(written != 12)
    {
      *errString = "written rtp file byte number error 1";
      fclose (fp);
      return fileLen;
    }

  written = fwrite (data, 1, dataLen, fp);
  fileLen += written;
  if(written != dataLen)
    {
      *errString = "written rtp file byte number error 2";
      fclose (fp);
      return fileLen;
    }

  if(csrc != NULL)
    {
      //write csrc
    }
  fclose (fp);
  return fileLen;
}

