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

//#define READ_FILE_BUFFER_SIZE 1024

using std::cout;
using std::cin;
using std::endl;


const int READ_FILE_BUFFER_SIZE = 1024;

/**
 * @brief This is a function for converting csvid file into a set of rtp files, each rtp file is a rtp packet.
 * They will be put into a folder named based on the time that the program is run at. 
 * Basically, it reads from a coded file, and then
 * write every CodeElement into a rtp file.
 * It will also generate a simple packetLen.txt file with the following information
 *  packetID   packetLength
 *  This information is here for futher manipulating the files, or inserting errors. 
 * @param filename The file name to read from 
 * @param folderPre The folder that the rtp files folder will be created under 
 * @param errString For writing the errors
 * @return no return value
 */
void
CS2RTP (const char* filename, const char* folderPre, const char** errString)
{

  const char payloadType = 98;
  uint32 numRead = 0;
  int ii = 0;
  const char* lenFileName = "packetLen.txt";

  //this records how many bytes left in one call of readOneCEBuffer, 
  //for the next fread, we only read BUFFER_SIZE - left, start at buffer+left
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

  FILE* fp = fopen (filename, "rb");
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
  uint16 ceSNo = 0;
  uint32 timeAll = time (NULL);

  srand (timeAll);
  uint16 randomBase = rand () % 60000;
  cout << "randomBase =" << randomBase << endl;
  assert (randomBase >= 0);
  /*
    char timeBuffer[20];
    sprintf(timeBuffer, "%d", timeAll); 
   */

  //extract the fileName from the full path
  string lastName = string (filename);
  lastName = lastName.substr (lastName.find_last_of ("\\/") + 1, lastName.length ());

  string folderName = string (folderPre) + "/" + lastName + ".rtpFN";
  checkC (folderName);

  string lenFileFinal = folderName + "/" + string (lenFileName);
  FILE* wp = fopen (lenFileFinal.c_str (), "w");


  if(wp == NULL)
    {
      *errString = "len txt file create error\n";
      fputs (*errString, stderr);
      return;
    }
  cout << "The rtp files will be generated to the following folder:\n";
  cout << folderName << endl;


  rtpP = new RtpPacket (payloadType, 1121, MAX_RTP_DATA_LEN);

  //char readOneCEBuffer(char* buffer, uint32* bufferSizeP, char* bufferSizeChanged, char** bufferP, CodeElement* ce, char* gotOne, uint32* left,   char** errString)
  while(numRead = fread (buffer + left, sizeof (char), bufferSize - left, fp))
    {

      CodeElement ceS;
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
          //	printf("here %d %d \n", isEndOfBuffer, gotOne);

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
              ceSNo++;
              //print the ce
              uint32 key = ceS.getKey ();
              //	printf("key = %ld, length =%ld\t", ceS.getKey(), ceS.getLength());
              if(key == 0)
                {

                  if(tl != NULL) delete tl;
                  tl = new TypeList (ceS);
                  //	tl->dumpTypes();
                }
              //printf("type = %s \n", (tl->getTypeFromKey(ceS.getKey())).c_str());
              //
              //need to write key and length to rtp packet too
              char* rtpBuffer = (char*) malloc (ceS.getLength () + 20);
              longInt keyLen[2];
              keyLen[0] = ceS.getKey ();
              keyLen[1] = ceS.getLength ();
              char* keyLenBuffer;
              size_t outputLen;
              keyLenBuffer = writeUInt (keyLen, 2, &outputLen, errString);
              memcpy (rtpP->getData (), keyLenBuffer, outputLen);
              memcpy (rtpP->getData () + outputLen, ceS.getData (), ceS.getLength ());
              rtpP->setDataLen (outputLen + ceS.getLength ());
              rtpP->setTimeStamp (time (NULL));
              rtpP->setSeqNo (ceSNo + randomBase);



              free (keyLenBuffer);

              rtpLen = rtpP->writeToFile (folderName, randomBase, errString);
              checkError (*errString);
              fprintf (wp, "%d  %d\n", ceSNo, rtpLen);

              /*

                 if(tl->getTypeFromKey(ceS.getKey()) == "UniformQuantizer")
                 {
                 UniformQuantizer* uq = new UniformQuantizer(ceS);
              //do sth
              //
              delete uq;
              }
               */
              //write into file
              //ceS.writeToFile(wfp, errString);
              //free the ce
              //free(ceS.data);
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
  delete rtpP;
  fclose (wp);
  delete tl;
  free (buffer);
  fclose (fp);

}
///@brief The help printing 

void
printHelp (char** argv)
{
  cout << "Name: " + string (argv[0]) << endl;
  cout << "Usage: " + string (argv[0]) << " [input file name] [parent folder that contains the folder of rtp Files]" << endl;
  cout << "             --specify the input file name and output file folder name as the arguments." << endl;
  cout << "             --The folder name will be based on [input file name]+.rtpF. The folder name will be printed to standard output." << endl;
  cout << "             --the parent folder must exist. The rtpF folder will be created by the program." << endl;
  cout << "             --There will be a folder based on the current time generated. The folder name will be printed to standard output." << endl;
  cout << "       " + string (argv[0]) << " -" << endl;
  cout << "             --reading the input file name from standard input." << endl;
  cout << "       " << "**You should use absolute path name, or the relevent path based on the location where the program is run." << endl;
}

/**
 * @brief The main test function
 */
int
main (int argc, char** argv)
{
  string inName, outName;
  if(argc == 1)
    {
      printHelp (argv);
      return 0;
    }
  else if(argc == 3)
    {
      inName = argv[1];
      outName = argv[2];

      //the *secret* testing command
      if(!(strcmp (argv[1], "-") || strcmp (argv[2], "-")))
        {
          inName = string ("../../../output/20140606_1719/1.case/news_cif_300.csvid");
          outName = string ("../../../output/20140606_1719/1.case");
          cout << "using1:" << inName << endl;
          cout << "output " << outName << endl;
        }

    }
  else if(argc == 2)
    {
      if(string (argv[1]) != "-")
        {
          cout << "only one argument, but not -, please read the help" << endl;
          printHelp (argv);
          return 0;
        }
      else
        {
          cout << "Please input the file path and name to read from:" << endl;
          cin>>inName;
          cout << "Please input the file path and name to write to:" << endl;
          cin>>outName;
        }
    }
  else
    {
      cout << "Invalid number of argumesnts, please read the help" << endl;
      printHelp (argv);
      return 0;
    }


  const char* err = NULL;
  //char* filename = "/home/jianwel/compsens/releases/output/20140606_1719/1.case/foreman_cif_300.csvid";
  //char* wName= "/home/jianwel/compsens/releases/output/20140606_1719/1.case/foreman_ljw.csvid";
  //char* filename = "/home/jianwel/compsens/releases/output/20140606_1719/1.case/part.csvid";
  CS2RTP (inName.c_str (), outName.c_str (), &err);
  checkError (err);
}



