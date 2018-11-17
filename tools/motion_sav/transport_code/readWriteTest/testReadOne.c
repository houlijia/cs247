#include "readUInt.h"
#include <stdio.h>

int
main ()
{
  unsigned char inputArray[4] = {11, 232, 33, 34};
  printf ("%d\n", sizeof (inputArray));
  longInt num = readOneUInt (inputArray, 1, 2, NULL);
  printf ("num = %ld\n", num);

}
