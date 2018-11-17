#ifdef __cplusplus
extern "C"
{
#endif

#include "readSInt.h"
#ifdef __cplusplus
}
#endif


#include <iostream>
using namespace std;

int
main ()
{
  char inputArray[5] = {11, 64, 232, 33, 34};
  SlongInt re = readOneSInt (inputArray, 1, 3, NULL);
  cout << re << endl;
}
