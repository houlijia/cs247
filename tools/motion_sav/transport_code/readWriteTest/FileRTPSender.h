/* 
 * File:   FileRTPSender.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 21, 2014, 3:19 PM
 */

#ifndef FILERTPSENDER_H
#define	FILERTPSENDER_H

#include "RTPSender.h"
#include <cstdio>

class FileRTPSender :RTPSender
{
public:
 /**
  *@param folderName The name of folder that the files will be written to. 
  * If it exists, print an error message and exit
  *  
  */
 FileRTPSender (const string& folderName0, bool isStd0) :folderName(folderName0),
   isStd(isStd0), lenFileName("packetLen.txt"), packetNo(0), wlp(NULL), wp(NULL)
 {
  string lenFileFinal = folderName + "/" + string (lenFileName);
   wlp = fopen (lenFileFinal.c_str (), "w");


  if(wlp == NULL)
    {
      const char* errString = "len txt file create error\n";
      fputs (errString, stderr);
      return;
    }
   
   if(isStd)
   {
   wp = stdout;
   }


 }
 void sendMeta(uint16 rtpSeqNo, uint8 QNo, longInt startPos, uint16 rtpLen)
 {
  
    fprintf (wlp, "%d %d %lu %d\n", rtpSeqNo, QNo, startPos, rtpLen);
 }

 int send(const RtpPacket* const rtp, const char** errString)
 {
  
 if(!isStd) 
   {
     string finalFileName = folderName + "/" + intToString(packetNo, 5) + ".rtp";
     wp = fopen(finalFileName.c_str () , "wb"); 
   }
 int written =0;
 int fileLen =0;

  written = fwrite (rtp->getReadOnlyHeader(), 1, 12, wp);
  fileLen +=written;
if(written != 12)
    {
      *errString = "written rtp file byte number error 1";
      if(!isStd)
{
        fclose (wp);
wp = NULL;
}
      return fileLen;
    }

  written = fwrite (rtp->getReadOnlyData(), 1, rtp->getDataLength(), wp);
  fileLen +=written;
  if(written != rtp->getDataLength())
    {
      *errString = "written rtp file byte number error 2";
      if(!isStd)
{
        fclose (wp);
wp =NULL;
}
      return fileLen;
    }


  
  if(!isStd)
   {
   fclose(wp);
wp =NULL;
   }

  packetNo++;
  return written;
  
 }
// FileRTPSender (const FileRTPSender& orig);
 
 virtual ~FileRTPSender ()
 {
   if(wlp!=NULL)
   {
     fclose(wlp);
     wlp = NULL;
   }
   if(!isStd && wp!=NULL)
   {

     fclose(wp);
     wp = NULL;
   }
 }
private:
 int packetNo;
  const string lenFileName;
 string folderName;
 bool isStd;
  FILE* wp;
FILE* wlp ;
};

#endif	/* FILERTPSENDER_H */

