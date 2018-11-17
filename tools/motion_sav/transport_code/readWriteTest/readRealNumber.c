#include "readRealNumber.h"
#include "readSInt.h"
#include <math.h>

double
readOnlyOneRealNumber (char* inputArray, uint32* bytesRead, const char** errString)
{
  int endP;
  SlongInt f = readOnlyOneSInt (inputArray, 0, &endP, errString);
  SlongInt e = readOnlyOneSInt (inputArray, endP + 1, &endP, errString);
  *bytesRead = endP + 1;
  /*
   *I do not think this will work
    if(e>0)
    {
      if(f>0)
       return f>>abs(e);
      else
       return -1 * (abs(f) >> abs(e));
    }
    else
      if(f>0)
       return f<<abs(e);
      else
       return -1 * (abs(f) << abs(e));
    }
   */
  return f * pow (2, e);
}


