#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <time.h>
#include "CodeElement.h"
#include "TypeList.h"
#include "RtpPacket.h"
#include "UniformQuantizer.h"
#include "Folder.h"
#include "writeUInt.h"
#include "readUInt.h"
#include "DropList.h"
#include "UdpRTPSender.h"
#include "FileRTPSender.h"
#include "unistd.h"
#include "CmdLineFind.h"
#include "RawVideoInfo.h"
#include "CSEncParam.h"
#include "TimeCalc.h"
#include "Stopwatch.h"


//#define READ_FILE_BUFFER_SIZE 1024

using std::cout;
using std::cin;
using std::endl;
using std::cerr;


const int READ_FILE_BUFFER_SIZE = 1024;
enum SendingMethodType { SEND_FILE, SEND_UDP, SEND_STD};

static inline longInt
MAX (longInt a, longInt b)
{
 return ((a > b) ? a : b);
}

static inline longInt
MAX_TR (longInt a, longInt b, longInt c)
{
 longInt mid = ((a > b) ? a : b);
 return ((mid > c) ? mid : c);
}

/**
 * @brief This is a function for converting csvid file into a set of rtp files, each rtp file is a rtp packet.
 * They will be put into a folder named based on the input csvid file name.
 *
 * The current packet format is like the following:
 * ----------------------------------
 *   packet header
 * --------------------------------
 *  Block Number (0-255)
 *  ---------------------------
 *  Small Side Info Code Elments
 *  ------------------------------
 *  CodeElement changed from Meta data part of a measurement block
 *  --------------------------------
 *  offset (of number of integers)
 *  --------------------------------
 *  Data
 *  ------------------------------- 
 *
 * It will also generate a simple packetLen.txt file with the following information
 *  packetID   packetLength
 *  This information is here for futher manipulating the files, or inserting errors. 
 * @param filename The file name to read from 
 * @param folderPre The folder that the rtp files folder will be created under 
 * @param errString For writing the errors
 * @return no return value
 */
void
CS2RTP (const char* filename, const string& folderName, uint32 minSplit, uint32 maxLen, bool useDrop, SendingMethodType sendingMethod, bool isFullSpeed, const string& ipString, const int& port, const char** errString)
{
 
 //extract the fileName from the full path
 //string lastName = string (filename);
 //lastName = lastName.substr (lastName.find_last_of ("\\/") + 1, lastName.length ());
 
 //string folderName = string (folderPre) + "/" + lastName + ".rtpF";
 
 
 
 
 const char payloadType = 98;
 uint32 numRead = 0;
 RawVideoInfo* rv =NULL;
 CSEncParam* ep =NULL;
 TimeCalc* tcalc =NULL;
 
 
 //  int ii = 0;
 
 
#if 1
 FILE* dropFp;
 dropFp = fopen("droplist.txt", "r");
 DropList dl(dropFp);
#endif
 
//UniformDropList dl(10);
 
 //this records how many bytes left in one call of readOneCEBuffer, 
 //for the next fread, we only read BUFFER_SIZE - left, starting at buffer+left
 int left = 0;
 
 //we need to record the oldBufferSize because we may change the bufferSize in the middle, and need to read some more to 
 //the resized buffer, and get a complete CE
 uint32 bufferSize, oldBufferSize;
 
 char* buffer = (char *) malloc (READ_FILE_BUFFER_SIZE * sizeof (char));
 if(buffer == NULL)
  {
   *errString = "malloc in readFile failed";
   return;
  }
 
 bufferSize = READ_FILE_BUFFER_SIZE;
 oldBufferSize = bufferSize;
 
 FILE* fp= fopen (filename, "rb");
 if(fp == NULL)
  {
   *errString = "no input file found\n";
   fputs (*errString, stderr);
   return;
  }
 
 
 char isEndOfBuffer = FALSE;
 char gotOne = FALSE;
 char bufferSizeChanged = FALSE;
 
 char* updatingBuffer = buffer;
 
 char** updatingBufferP = &updatingBuffer;
 TypeList* tl = NULL;
 RtpPacket* rtpP;
 
 uint8 QNo = 0;
 uint16 rtpSeqNo = 0;
 uint32 timeAll = time (NULL);
 
 srand (timeAll);
 uint16 randomBase = rand () % 60000;
 //	cout<<"randomBase ="<<randomBase<<endl;
 assert (randomBase >= 0);
 
 /*
  char timeBuffer[20];
  sprintf(timeBuffer, "%d", timeAll); 
  */
 
 FileRTPSender* fileSender = NULL;
 
 if(sendingMethod == SEND_FILE)
  {
   fileSender = new FileRTPSender(folderName, false);
  }
 
 
 
 CodeElement basicCS[3];
 CodeElement CodeRegion;
 
 //matrix wh and uniform quantizer
 CodeElement MatrixAndQuan[2];
 
 CodeElement ceS;
 
 rtpP = new RtpPacket (payloadType, 1121, maxLen);
 uint32 leftBytes;
 
 uint32 averageLeft;
 uint32 praticalMaxLen = maxLen;
 uint32 nSplit;
 uint32 metaDataEnd = 0;
 uint32 headerLen = 0;
 uint32 metaLen[3];
 
 UdpRTPSender* udpSender;
 
 if(sendingMethod == SEND_UDP)
  {
   udpSender = new UdpRTPSender(ipString, port);
   udpSender->init();
   
  }

  Stopwatch timer;
  uint32 nrtp;
 
 
 
 //char readOneCEBuffer(char* buffer, uint32* bufferSizeP, char* bufferSizeChanged, char** bufferP, CodeElement* ce, char* gotOne, uint32* left,   char** errString)
 while(numRead = fread (buffer + left, sizeof (char), bufferSize - left, fp))
  {
   isEndOfBuffer = FALSE;
   
   
   if(numRead < bufferSize - left)
    {
     bufferSize = numRead + left;
    }
   
   left = 0;
   
   while(!isEndOfBuffer)
    {
     isEndOfBuffer = ceS.readOneCEFromBuffer (buffer, &bufferSize, &bufferSizeChanged, updatingBufferP, &gotOne, &left, errString);
     checkError (*errString);
     
     //if buffer size changed, we want the left = 0, so that it can call the readOneCEBuffer again. should return TRUE
     if(bufferSizeChanged)
      {
       buffer = *updatingBufferP;
       //assert(bufferSize > oldBufferSize);
       fread (buffer + oldBufferSize, 1, bufferSize - oldBufferSize, fp);
       oldBufferSize = bufferSize;
       bufferSizeChanged = FALSE;
       
       //this =0 maybe extra, but make sure left =0
       left = 0;
      }
     if(gotOne)
      {
       uint16 rtpLen;
       //print the ce
       uint32 key = ceS.getKey ();
       if(key == 0)
        {
         //if multiple tl received, maybe impossible
         if(tl != NULL) delete tl;
         tl = new TypeList (ceS);
        }
       string type = tl->getTypeFromKey (ceS.getKey ());
       //cout<<type<<endl;
       
       if(type == "VidRegion")
        {
         clone_CE (&ceS, &CodeRegion, errString);
        }
       else if(type == "SensingMatrixWH" || type == "UniformQuantizer")
        {
         
         char pos = 0;
         if(type == "UniformQuantizer")
          pos = 1;
         clone_CE (&ceS, &MatrixAndQuan[pos], errString);
        }
       else if(type == "QuantMeasurementsBasic")
        {
         int endP;
         int bytes=0;
          nrtp =0;
         
         
         if(sendingMethod == SEND_UDP && !isFullSpeed) 
          {
           assert(rv!=NULL && ep!=NULL);
           tcalc = new TimeCalc(rv, ep);
          }
         //uint8 pos = findPos((const unsigned char*)(ceS.getData ()), 0, 3, 30, errString);
         
         longInt nbins = readOnlyOneUInt(ceS.getData()+bytes , 0, &endP, errString);
         bytes += endP+1;
         longInt n_noclip = readOnlyOneUInt(ceS.getData()+bytes , 0, &endP, errString);
         bytes += endP+1;
         longInt len_b= readOnlyOneUInt(ceS.getData()+bytes , 0, &endP, errString);
         bytes += endP+1;
         longInt len_s = readOnlyOneUInt(ceS.getData()+bytes , 0, &endP, errString);
         bytes += endP+1;
         assert(len_s ==0);
         //cerr<<"n_noclip="<<n_noclip<<endl;
         
         
         //find the end of the meta data, metaDataEnd +1 is the start of data part
         metaDataEnd = findPos ((unsigned char*) (ceS.getData ()), 0, 8+n_noclip, ceS.getLength (), errString);
         
         //set the data part to the realData ptr in the CodeElment
         //reset the length, so that we can call ceS.WriteToBuffer(rtpP->getData(), errString)
         ceS.calcRealDataPos (metaDataEnd + 1);
         
         uint32 nIntBeforeSend = countNumbers((const unsigned char*)ceS.getRealData(), ceS.getRealLength());
//         cout<<"nIntBeforeSend" << QNo<<"  :"<<nIntBeforeSend<<endl; 
         
         //only the length of the data part
         leftBytes = ceS.getRealLength ();
         
         assert (maxLen >= 500);
         uint32 bytesW = 0;
         size_t oneTime = 0;
         longInt startPos = 0;
         uint32 offset = 0;
         bool isFirstPacket;
         int estimatedNRtp; 
         int interval =0;
         //the exact maximum packet len
         //the 3 bytes at the end counts the QNo, and offset at the beginning of each data part
         uint32 maximumPacketLen = MAX_TR (basicCS[0].getAllLen (errString), basicCS[1].getAllLen (errString), basicCS[2].getAllLen (errString)) + CodeRegion.getAllLen (errString) + MAX (MatrixAndQuan[0].getAllLen (errString), MatrixAndQuan[1].getAllLen (errString)) + metaDataEnd + 1 + rtpP->getHeaderLength () + leftBytes / minSplit + 3;
         if(maximumPacketLen < maxLen)
          {
           maxLen = maximumPacketLen;
          }
         
         
         while(leftBytes > 0)
          {
           bytesW = 0;
           oneTime = 0;
           
           startPos = ceS.getRealLength () - leftBytes;
           
           
           //first write the block number byte, so that the receiver can get this byte first, and make a decision on whether it should continue reading. 
           memcpy (rtpP->getData (), &QNo, 1);
           bytesW += 1;
           
           
           oneTime = basicCS[rtpSeqNo % 3].writeToBuffer (rtpP->getData () + bytesW, errString);
           
           bytesW += oneTime;
           
           oneTime = CodeRegion.writeToBuffer (rtpP->getData () + bytesW, errString);
           
           bytesW += oneTime;
           
           oneTime = MatrixAndQuan[rtpSeqNo % 2].writeToBuffer (rtpP->getData () + bytesW, errString);
           
           bytesW += oneTime;
           
           
           oneTime = ceS.writeToBuffer (rtpP->getData () + bytesW, errString);
           bytesW += oneTime;
           
           if(offset==0)
            isFirstPacket=true;
           
           
           if(dl.isInside(rtpSeqNo) && useDrop)
            {
              cerr<<"calling this drop writting"<<endl;
              oneTime = ceS.writeDataToBufferDrop (rtpP->getData () + bytesW, maxLen - rtpP->getHeaderLength () - bytesW, QNo, &offset, startPos, &leftBytes,   errString);
            }
           else
            {
             oneTime = ceS.writeDataToBuffer (rtpP->getData () + bytesW, maxLen - rtpP->getHeaderLength () - bytesW, QNo, &offset, startPos, &leftBytes, errString);
            }
           
           bytesW += oneTime;
           nrtp++;
           
           rtpP->setDataLen (bytesW);
           rtpP->setTimeStamp (time (NULL));
           rtpP->setSeqNo (rtpSeqNo + randomBase);
           
           if(sendingMethod == SEND_FILE)
            {
             //                      rtpLen = rtpP->writeToFile (folderName, randomBase, errString);
             rtpLen = fileSender->send(rtpP, errString);
             fileSender->sendMeta(rtpSeqNo, QNo, startPos, rtpLen);
             checkError (*errString);
            }
           else if(sendingMethod == SEND_UDP)
            {
             rtpLen = udpSender->send(rtpP, errString);
             //if(isFirstPacket && !isFullSpeed)
              {
              estimatedNRtp = ceil(ceS.getRealLength() / oneTime);
               //cerr<<"estimatedNrtp"<<estimatedNRtp<<endl;
//exit(0);
               interval = tcalc->getInterval(estimatedNRtp);
               //isFirstPacket = false;

              }

               cerr<<"sleep interval="<<interval<< " "<<nrtp<<endl;
               usleep(interval);
               //usleep(500);
             //cerr<<"sending udp "<<rtpLen<<endl;
             checkError (*errString);
             
            }
           
           
           rtpSeqNo++;
           if(rtpSeqNo > 65535) rtpSeqNo = 0;
          }
         
         QNo++;
         if(QNo > 255) QNo = 0;
         
         
        }
       else if(type == "Type List" ||
               type == "CS_EncParams" ||
               type == "RawVidInfo")
        {
         char pos = 0;
         if(type == "Type List");
         
         else if(type == "CS_EncParams")
          {
           pos = 1;
           ep = new CSEncParam(ceS);
          }
         else if(type == "RawVidInfo")
          {
           rv = new RawVideoInfo(ceS);
           pos = 2;
          }
         
         clone_CE (&ceS, &basicCS[pos], errString);
         
        }
       else
        {
         string error = "type error:" + type;
         //?? will the char be freed after the function exit
         *errString = error.c_str ();
         fputs (*errString, stderr);
         return;
        }
       
       
       gotOne = FALSE;
      }
     
     //if left >0, we must set isEndOfBuffer to TRUE
     assert (left >= 0);
     if(left > 0)
      {
       //memmove(buffer, buffer + bufferSize - left, left);
       //use memmove here instead of memcpy, because I think the two mem spaces might overlap
       memmove (buffer, *updatingBufferP, left);
       *updatingBufferP = buffer;
      }
     
    }
   
  }

cout<<"time cost = "<<timer.read()<<" seconds"<<endl;

 delete rtpP;

if(!isFullSpeed)
{
  delete tcalc;
}
 delete tl;
 free (buffer);

  if(fp!= NULL)
{
      fclose (fp);
fp = NULL;
}
 
 if(sendingMethod == SEND_UDP && udpSender !=NULL)
  {
   delete udpSender;
   udpSender = NULL;
  }
 if(fileSender !=NULL)
  {
   delete fileSender;
   fileSender = NULL;
  }
 
}
///@brief The help printing 

void
printHelp (const CmdLineFind* clfp, char** argv)
{
 //cout << "Name: " + string (argv[0]) << endl;
 cout<<"Generate RTP packets from a compressive sensing measurement file (.csvid):"<<endl;
 string tag = "";
 
 cout << string (argv[0]) << "[-send <[file, udp, std]>] [-input <input file name>] [-output <output folder name>] [-minSplit <minium packet splitting amount>] [-maxLen <maximum packet length>] [-isFullSpeed <0,1>] [-ip [<ipaddress>] -port[<port number>]" << endl;
 
 clfp->usage (tag);
 cout<<endl<<"EXAMPLE: "<<endl;
 cout<<string(argv[0])<<" "<<"-send file -input news.csvid -output outputFolder"<<endl; 
 cout<<"This example reads from news.csvid, and outputs it to a folder named 'outputFolder'. If that folder exists, a warning will be printed, and the program will exit. For the minSplit and maxLen argument, the program will use the default value listed in the table above";
} 
/**
 * @brief The main test function
 */
int
main (int argc, char** argv)
{
 CmdLineFind clf( argc, argv );
 
 const char* err = NULL;
 string inName, outName;
 uint32 minSplit, maxLen;
 SendingMethodType sendingMethod ;
 string ipString;
 int port;
 bool isFullSpeed =true;
 
 string sendMethodS = clf.find("-send", "file", "This tag specifies where to write the rtp packets. Options: <file, udp, std>");
 cout<<"using "<<sendMethodS<<endl;
 if(sendMethodS == "file")
  {
   sendingMethod = SEND_FILE; 
  }
 else if(sendMethodS == "udp") 
  {
   sendingMethod = SEND_UDP;
  }
 else if(sendMethodS == "std")
  {
   sendingMethod = SEND_STD;
  }
 minSplit =clf.find("-minSplit", 12, "This can specify the minimum number of packets a block will be split");
 maxLen =clf.find("-maxLen", 512, "This can specify the max length of the RTP packet ");
 
 inName = clf.find("-input","../../../output/20140623_1041/1.case/news_cif_300.csvid", "This is the input csvid file name");

  {
   string defaultOutName = inName+".rtpF";
   
   outName =clf.find("-output", defaultOutName, "This is the output folder name. If it exists, the program will return after a warning");

 if(sendingMethod == SEND_FILE)
{
   //check if out Name exist
   checkC (outName);
   cout << "The rtp files will be generated to the following folder:\n";
   cout << outName<< endl;
}
  }
// else if(sendingMethod == SEND_UDP)
   
   ipString = clf.find("-ip", "127.0.0.1", "This is the IP that it will send to");
   port = clf.find("-port", 9003, "This is the port it will send to");
   isFullSpeed = clf.find("-isFullSpeed", false, "This controls whether we send with full speed. If full speed, no delay is used");
   cout<<"toIP "<<ipString<<endl;

bool useDrop = clf.find("-useDrop", 0, "This controls whether we drop some packets by setting the content to bins+2");
 
 if(argc == 1)
  {
   printHelp (&clf, argv);
   return 0;
  }
 
 
 
 CS2RTP (inName.c_str (), outName.c_str (), minSplit, maxLen, useDrop, sendingMethod, isFullSpeed, ipString, port, &err);
 checkError (err);
}



