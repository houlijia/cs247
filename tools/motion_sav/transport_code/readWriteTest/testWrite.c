#include "writeUInt.h"
#include <stdio.h>

int
main ()
{
  longInt testArray[2] = {127, 13345};
  //	longInt testArray[1] = {13345};
  char* output = NULL;
  char* initError = "";
  char** errString = &initError;
  size_t outputL;
  uint32 j;

  checkError ();

  output = writeUInt (testArray, 2, &outputL, errString);


  printf ("\noutput in main:");
  printf ("outputL = %d\n", outputL);
  //debugging, will use my debugging class later
  for(j = 0; j < outputL; j++)
    printf ("%d ", (unsigned char) output[j]);
  printf ("\n");


  //free ouput
  if(output != NULL)
    free (output);

}
