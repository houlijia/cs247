#include "common_ljw.h"
#include <stdio.h>

/**
 * @brief A utility function that check a given errString pointer
 * If the errString is not null, print it out
 * @param errString -The error string to check
 */
void
checkError (const char* errString)
{
  if(errString != NULL)
    {
      fputs ("checked error (ljw):", stderr);
      fputs (errString, stderr);
      fputs ("\n", stderr);
      exit (0);
    }
}





