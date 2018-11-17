/*
 * =====================================================================================
 *
 *       Filename:  Folder.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/25/2013 03:50:15 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Jianwei Liu (ljw), ljw725@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef  folder_INC
#define  folder_INC

#include <stdio.h>
#include <sys/stat.h>
#include <string.h>
#include <sstream>



using std::string;

inline bool
my_checkFolder (const string& fname)
{
  struct stat sb;
  bool ise = stat (fname.c_str (), &sb);
  return !ise;
}

/**
 * @brief This is a function for checking whether a folder 
 * exist, if it does not exist, create it. But the assumption
 * is its parent folder exists.
 */

inline void
checkC (const string& fname)
{
  int a = 0;
  if (!my_checkFolder (fname))
    {
      a = mkdir ((fname).c_str (), 0777);
    }
  if (a < 0)
    {

      std::cerr << "open dic error " << fname << std::endl;
      //	printf( "Error opening file: %s\n", strerror(errno ) );
    }
}

#endif   /* ----- #ifndef folder_INC  ----- */
