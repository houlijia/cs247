/* 
 * File:   FileRTPReceiver.h
 * Author: Jianwei Liu<jianwel@g.clemson.edu>
 *
 * Created on July 21, 2014, 4:45 PM
 */

#ifndef FILERTPRECEIVER_H
#define	FILERTPRECEIVER_H

#include "RTPReceiver.h"
#include "DropList.h"

class FileRTPReceiver : public RTPReceiver
{
public:
 FileRTPReceiver (const string& inputFolder0): fp(NULL), inputFolder(inputFolder0), rtpNo(0)
 {
   
 };


 RtpPacket* getPacket(const char** errString)
 {
  RtpPacket* rtpP;

   if(dlp !=NULL && dlp->isInside(rtpNo))
    {
     cerr<<"####dropping "<<rtpNo<<endl;
      rtpNo ++;
      //
      return NULL;
    }
   

   int readRtp = rtpNo;

  //you can open this code later to simulate packet dropping
#if 0
   int interval = 22;
   const int totalRtp = 323;
   int endPos = totalRtp / (interval*2) * (interval*2);
   if(rtpNo % interval ==0 && rtpNo % (interval*2) !=0 && rtpNo!=0 && rtpNo <=endPos  )
    {
     readRtp += interval;
    }
   if(rtpNo % (interval*2) ==0 && rtpNo!=0 && rtpNo <= endPos)
    {
     readRtp -= interval;
    }

#endif
   //sprintf(numberStr, "%05d", rtpNo);
   string inputFileName = string(inputFolder) + "/" +intToString(readRtp, 5) + ".rtp";
   fp = fopen(inputFileName.c_str(), "rb");
   if(fp == NULL)
    {
     cout<<"NULL file stopped: "<<inputFileName<<endl;
      *errString = "NoFile";
     return NULL;
    }
   else
    {
     cout<<inputFileName<<endl;
    }
   rtpP = new RtpPacket (fp);
   fclose (fp);
   rtpNo++;
   return rtpP;

 }
 //FileRTPReceiver (const FileRTPReceiver& orig);
 virtual ~FileRTPReceiver () {
  if(dlp != NULL)
   {
    delete dlp;
    dlp = NULL;
   }
 };
private:
 int rtpNo;
 FILE* fp;
 const string inputFolder;

};

#endif	/* FILERTPRECEIVER_H */

