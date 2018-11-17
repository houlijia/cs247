#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <assert.h>
#include <string>
#include <iostream>

#include "RtpPacket.h"
#include "Folder.h"


using std::cout;
using std::cin;
using std::endl;

/**
 * @brief This is a function for converting csvid file into a set of rtp files, each rtp file is a rtp packet.
 * They will be put into a folder named based on the time that the program is run at. 
 * Basically, it reads from a coded file, and then
 * write every CodeElement into a rtp file.
 * It will also generate a simple packetLen.txt file with the following information
 *  packetID   packetLength
 *  This information is here for futher manipulating the files, or inserting errors. 
 * @param outName The file name to write to
 * @param inFolder The folder that the rtp files folder will be read from 
 * @param errString For writing the errors
 * @return The number of rtp files read 
 */
uint32
RTP2CS (const char* inFolder, const char* outName, const char** errString)
{
  FILE* fp;
  RtpPacket* rtpP;
  FILE* wp = fopen (outName, "wb");

  if(wp == NULL)
    {
      *errString = "out cs file create error in RTP2CS\n";
      fputs (*errString, stderr);
      return 0;
    }

  string inFileName;
  uint32 i = 1;

  while(1)
    {
      inFileName = string (inFolder) + "/" + convertInt (i) + ".rtp";
      fp = fopen (inFileName.c_str (), "rb");
      if(fp == NULL && i == 1)
        {
          *errString = "There is no file in the input folder\n";
          fputs (*errString, stderr);
          return 0;
        }
      if(fp == NULL) break;
      rtpP = new RtpPacket (fp);
      fclose (fp);
      uint32 written = fwrite (rtpP->getData (), 1, rtpP->getDataLength (), wp);
      i++;
      if(written != rtpP->getDataLength ())
        {
          *errString = "written bytes number does not match in RTP2CS\n";
          fputs (*errString, stderr);
          return i - 1;
        }
      delete rtpP;

    }
  fclose (wp);
  return i - 1;

}
///@brief The help printing 

void
printHelp (char** argv)
{
  cout << "Name: " + string (argv[0]) << endl;
  cout << "Usage: " + string (argv[0]) << " [input folder name] [output file name]" << endl;
  cout << "             --specify the input folder name and output file folder name as the arguments." << endl;
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
          inName = string ("../../../output/20140606_1719/1.case/news_cif_300.csvid.rtpF");
          outName = string ("../../../output/20140606_1719/1.case/news_cif_300.around.csvid");
          cout << "input:" << inName << endl;
          cout << "output: " << outName << endl;
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
          cout << "Please input the folder to read from:" << endl;
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

  RTP2CS (inName.c_str (), outName.c_str (), &err);
  checkError (err);
}



