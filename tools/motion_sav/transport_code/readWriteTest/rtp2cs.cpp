#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <assert.h>
#include <string>
#include <iostream>
#include "RTP2CSFactory.h"
#include "FileBufferSender.h"
#include "TcpBufferSender.h"
#include "RtpPacket.h"
#include "FileRTPReceiver.h"
#include "UdpRTPReceiver.h"
#include "CmdLineFind.h"

using std::cout;
using std::cin;
using std::endl;

enum RecvMethodType{RECV_UDP, RECV_FILE};
enum SendMethodType{SEND_TCP, SEND_FILE};
enum DropMethodType{ DROP_NO, DROP_UNIFORM, DROP_FROM_TXT};




/**
 * @brief This is a function for testing the read and write functions. 
 * Basically, it reads from a coded file, and print the key, length of the CodeElement, and then
 * write them back into a file. 
 * @param nputFolder The file name to read from 
 * @param writeName The file name to write to 
 * @param errString For writing the errors
 * @return no return value
 */
void
RTP2CS (const char* inputFolder, const char* writeName,  RecvMethodType recvM, const string& ipRecv, const string& portRecv, const string& ipSend, const string& portSend,  SendMethodType sendM, DropMethodType dropOption, int dropN,  const char** errString) 
{
 
 FILE* fp, *wfp;
 wfp = fopen(writeName, "wb");
 //wfp_global = wfp;
 
 
 BufferSender* fbs;
if(sendM = SEND_FILE)
 { 
  fbs = new FileBufferSender(100000, wfp);
 }
else
 {
  fbs = new TcpBufferSender(ipSend.c_str(), portSend.c_str(),10000);
 }
 RTP2CSFactory* rfp = new RTP2CSFactory(12, fbs);
 //rfp_global = rfp;
 RTP2CSFactory& rf = *rfp;
 //uint32 rtpNo = 0;
 RtpPacket* rtpP =NULL;
 RTPReceiver* receiver =NULL;
 
 int timeout = 3000; 
 
 string inputF(inputFolder);
 if(recvM == RECV_FILE)
  {
   cerr<<"using file"<<endl;
   receiver = new FileRTPReceiver(inputF);
   AbstractDropList* dl = NULL; 
   if(dropOption == DROP_UNIFORM)
    {
     dl = new UniformDropList(dropN);
     receiver->setDrop(dl);
    }
   else if(DROP_FROM_TXT == dropOption)
    {
     
     FILE* dropFp;
     dropFp = fopen("droplist.txt", "r");
     if(dropFp != NULL)
      {
       dl = new DropList(dropFp);
       fclose(dropFp);
      receiver->setDrop(dl);
      }
    }
  }
 else if(recvM == RECV_UDP)
  {
   cerr<<"using udp haha"<<endl;
   receiver = new UdpRTPReceiver(ipRecv, atoi(portRecv.c_str()), timeout);
   
  }
 //receiver_global = receiver;
 int packetNo =0; 
 //signal(SIGALRM, cleanOnExit);
 //char numberStr[10];
 while(1)
  {
   
   rtpP = receiver->getPacket(errString);


   if(rtpP ==NULL ){
   if(recvM == RECV_FILE )
    {

      if(*errString == "NoFile" )
       {
        *errString = NULL;
       break;
       }
      else
       continue;
    }
   else if(recvM == RECV_UDP)
    {
     if (packetNo == 0 )
      {
       cout << '.';
       continue;
      }
     break;
    }
    }//if NULL
 rf.insertRTPPacket(rtpP, rtpP->getTotalLength(), errString);
    packetNo++;
   cerr<<"@@packetNo "<<packetNo<<endl;
  delete rtpP;
 
  }//while
  if(receiver!=NULL)
  {
   delete receiver;
   receiver = NULL;
  }
 
  delete rfp;
  

 if(wfp !=NULL)
  {
   cerr<<"closing the write file"<<endl;
   fclose(wfp);
  }
 

 
}
///@brief The help printing 
#if 0
void
printHelp (char** argv)
{
 cout << "Name: " + string (argv[0]) << endl;
 cout << "Usage: " + string (argv[0]) << " <input folder name> <output file name>" << endl;
 cout << "             --specify the input and output names as the arguments. **" << endl;
 cout << "             --if the third argument is '#', the output csvid name will be <input folder name>.csvid" << endl;
 cout << "       " + string (argv[0]) << " # #" << endl;
 cout << "             --Using the default input and output file names specified in the code. For easier testing. It reads from ../../../output/20140623_1041/1.case/foreman_cif_300.csvid" << endl;
 cout << "       " << "**You should use absolute path name, or the relevent path based on the location where the program is run." << endl;
}
#endif 

void
printHelp (const CmdLineFind* clfp, char** argv)
{
 //cout << "Name: " + string (argv[0]) << endl;
 cout<<"Generate csvid file from RTP packets reading from local folder or network:"<<endl;
 string tag = "";
 
 cout << string (argv[0]) << "[-recv<[file, udp]>] [-ipFrom <ip adress>] [-portFrom <port number>] [-send <[file, tcp]>] [-input <input file name>] [-output <output folder name>] [-drop <no, uniform, fromFile>]" << endl;
 
 clfp->usage (tag);
 cout<<endl<<"EXAMPLE: "<<endl;
 cout<<string(argv[0])<<" "<<"-recv file -input inputFolderName -drop uniform -dropN 10 -output output.csvid"<<endl; 
 cout<<"This example reads from a folder called inputFolderName and outputs the csvid as output.csvid. It will drop using uniform packet dropper. It will drop one packet every dropN. dropN is specified as 10 in this example. If it is not specified, it is 50 by default. For other prameters, the program will use the default value listed in the table above"<<endl;

 cout<<string(argv[0])<<" "<<"-recv udp -ipFrom 192.168.0.1 -portFrom 9002 -output output.csvid"<<endl; 
 cout<<"This example reads RTP packets from a folder a UDP socket and outputs the csvid as output.csvid. For other prameters, the program will use the default value listed in the table above"<<endl;
cout<<endl;

} 

/**
 * @brief The main test function
 */
int
main (int argc, char** argv)
{
  CmdLineFind clf( argc, argv );
 const char* err = NULL;
 string ip;
 string port;

 string ipTo;
 string portTo;
 
 RecvMethodType recvM;
 SendMethodType sendM;
DropMethodType dropM; 
 
 if(globalError != NULL)
  {
   fputs (globalError, stderr);
   exit (0);
  }

  string recvMS = clf.find("-recv", "file", "This controls where the rtp packets are received");
  
  if(recvMS == "file")
   {
      recvM = RECV_FILE; 
   }
  else if(recvMS == "udp")
   {
    
      recvM = RECV_UDP; 
   }
 string inName = clf.find("-input","../../../output/20140623_1041/1.case/news_cif_300.csvid.rtpF","This is the input file name" );
 string outName = clf.find("-output","../../../output/20140623_1041/1.case/news.around.csvid", "This is the output file name" );
 
 string sendMS = clf.find("-send", "file", "This controls how we send the output csvid");
 if(sendMS == "file")
  {
sendM = SEND_FILE;
  }
 else if(sendMS =="tcp")
  {
   
sendM = SEND_TCP;
  }
 string dropMS = clf.find("-drop", "no", "This controls how we drop packets");
 if(dropMS =="no")
  {
dropM = DROP_NO;
  }
 else if(dropMS == "fromFile")
  {
   
dropM = DROP_FROM_TXT;
  }
 else if(dropMS =="uniform")
  {
   
dropM = DROP_UNIFORM;
  }
 //if(recvM == RECV_UDP)
  {
   ip = clf.find("-ipFrom", "127.0.0.1", "This is the IP address from which the RTP packets are read from");
  port = clf.find("-portFrom", "9003", "This is the port from which the RTP packets are read from");
  cout<<"from ip "<<ip<<endl;
   
  }
 //if(sendM == SEND_TCP)
  {
   ipTo = clf.find("-ipTo", "127.0.0.1", "This is the IP address to send to");
  portTo = clf.find("-portTo", "9003", "This is the port to send to");
   

   
  }
  int dropN = clf.find("-dropN", 50, "drop one packet out of dropN packets");
 
 if(argc == 1)
  {
   printHelp (&clf, argv);
   return 0;
  }
  
 
 RTP2CS (inName.c_str (), outName.c_str (),  recvM, ip, port,ipTo, portTo, sendM, dropM, dropN,  &err);
 checkError (err);
}



