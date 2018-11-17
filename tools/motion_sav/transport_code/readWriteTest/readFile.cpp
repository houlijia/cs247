#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <assert.h>
#include <string>
#include <iostream>
#include "CodeElement.h"
#include "TypeList.h"
#include "RtpPacket.h"
#include "UniformQuantizer.h"

//#define READ_FILE_BUFFER_SIZE 1024

using std::cout;
using std::cin;
using std::endl;


const int READ_FILE_BUFFER_SIZE = 1024;

/**
 * @brief This is a function for testing the read and write functions. 
 * Basically, it reads from a coded file, and print the key, length of the CodeElement, and then
 * write them back into a file. 
 * @param filename The file name to read from 
 * @param writeName The file name to write to 
 * @param errString For writing the errors
 * @return no return value
 */
void
readWriteFile (const char* filename, const char* writeName, bool isWrite, bool isReadStd, bool isWriteStd, const char** errString)
{
  uint32 numRead = 0;
  int ii = 0;

  //this records how many bytes left in one call of readOneCEBuffer, 
  //for the next fread, we only read BUFFER_SIZE - left, start at buffer+left
  int left = 0;

  //we need to record the oldBufferSize because we may change the bufferSize in the middle, and need to read some more to 
  //the resized buffer, and get a complete CE
  uint32 bufferSize, oldBufferSize;
  int nblock = 0;

  char* buffer = (char *) malloc (READ_FILE_BUFFER_SIZE * sizeof (char));
  if(buffer == NULL)
    {
      *errString = "malloc in readFile failed";
      return;
    }

  bufferSize = READ_FILE_BUFFER_SIZE;
  oldBufferSize = bufferSize;

  FILE* fp;
  if(!isReadStd)
    {
      fp = fopen (filename, "rb");



      if(fp == NULL)
        {
          *errString = "no input file found";
          return;
        }
    }
  else
    {
      fp = stdin;
    }

  FILE* wfp;
  if(!isWriteStd)
    {
      if(isWrite)
        {
          wfp = fopen (writeName, "wb");
          if(wfp == NULL)
            {
              *errString = "no output file can be created, maybe not enough space?";
              return;
            }
        }
    }
  else
    {
      wfp = stdout;
    }


  char isEndOfBuffer = FALSE;
  char gotOne = FALSE;
  char bufferSizeChanged = FALSE;

  char* updatingBuffer = buffer;

  char** updatingBufferP = &updatingBuffer;
  TypeList* tl = NULL;
  uint32 nRead = 0;
  uint32 nBlock = 0;

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
          CodeElement ceS;
          isEndOfBuffer = ceS.readOneCEFromBuffer (buffer, &bufferSize, &bufferSizeChanged, updatingBufferP, &gotOne, &left, errString);
          checkError (*errString);

          //if buffer size changed, we want the left = 0, so that it can call the readOneCEBuffer again. should return TRUE
          if(bufferSizeChanged)
            {
              buffer = *updatingBufferP;
              assert (bufferSize > oldBufferSize);
              fread (buffer + oldBufferSize, 1, bufferSize - oldBufferSize, fp);
              oldBufferSize = bufferSize;
              bufferSizeChanged = FALSE;

              //this =0 maybe extra, but make sure left =0
              left = 0;
            }
          if(gotOne)
            {
      		nRead++;
              left = 0;
              //print the ce
              longInt key = ceS.getKey ();
              longInt temp = ceS.getLength ();
              // The next statement produces a zero length in Cygwin 32 bit, for some unclear reason
              // printf("key = %ld, length=%lu ", ceS.getKey(),(unsigned long) ceS.getLength());
              printf ("key = %lu, ", ceS.getKey ());
              printf ("length = %lu \t,", temp);
              if(key == 0)
                {

                  if(tl != NULL) delete tl;
                  tl = new TypeList (ceS);
                  //	tl->dumpTypes();
                }
              string type = tl->getTypeFromKey (ceS.getKey ());
              printf (" type = %s \n", (type).c_str ());
              if(type == "QuantMeasurementsBasic")
                  nblock++;
                  
	    
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
              if(isWrite)
                ceS.writeToFile (wfp, errString);
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
        printf("number of CodeElemnts = %d\n", nRead);
        printf("#block = %d\n", nblock);
  delete tl;
  free (buffer);
  if(!isReadStd)
    fclose (fp);
  if(isWrite && !isWriteStd)
    fclose (wfp);

}
///@brief The help printing 

void
printHelp (char** argv)
{
  cout << "Name: " + string (argv[0]) << endl;
  cout << "Usage: " + string (argv[0]) << " [input file name] [output file name]" << endl;
  cout << "             --specify the input and output file names as the arguments. **" << endl;
  cout << "       " + string (argv[0]) << " [-] [-]" << endl;
  cout << "             --reading the csvid file from standard input." << endl;
  cout << "             --if the third argument is also '-', the output csvid will be directed to standard output." << endl;
  cout << "       " + string (argv[0]) << " # #" << endl;
  cout << "             --Using the default input and output file names specified in the code. For easier testing. It reads from ../../../output/20140623_1041/1.case/foreman_cif_300.csvid" << endl;
  cout << "       " << "**You should use absolute path name, or the relevent path based on the location where the program is run." << endl;
}

/**
 * @brief The main test function
 */
int
main (int argc, char** argv)
{
  if(globalError != NULL)
    {
      fputs (globalError, stderr);
      exit (0);
    }
  string inName = "";
  string outName = "";
  bool isWrite = false;
  bool isReadStd = false;
  bool isWriteStd = false;
  if(argc == 1)
    {
      printHelp (argv);
      return 0;
    }
  else if(argc == 3)
    {
      inName = argv[1];
      outName = argv[2];
      if(inName == "-")
        {
          isReadStd = true;
        }
      if(outName == "-")
        {
          isWriteStd = true;
        }
      //the *secret* testing command
      if(!(strcmp (argv[1], "#") || strcmp (argv[2], "#")))
        {
          inName = string ("../../../output/20140623_1041/1.case/foreman_cif_300.csvid");
          outName = string ("../../../output/20140623_1041/1.case/foreman_cif_ljw_300.csvid");
          cout << "using " << inName << endl;
          cout << "output " << outName << endl;
        }
      isWrite = 1;



    }
  else if(argc == 2)
    {
      inName = argv[1];
      if(inName == "-")
        {
          isReadStd = true;
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
  readWriteFile (inName.c_str (), outName.c_str (), isWrite, isReadStd, isWriteStd, &err);
  checkError (err);
}



