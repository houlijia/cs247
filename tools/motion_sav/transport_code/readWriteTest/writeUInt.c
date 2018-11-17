/**
 * @file writeUInt.c
 * @brief The file for basic  uint writing
 * @author Jianwei Liu, Bell Labs
 */


#include <stdlib.h>
#include <stdio.h>
#include "writeUInt.h"

/**
 * @brief This is a function for write only one int number.
 * @param number The number to be written
 * @param output The output buffer to write to
 * @param errString
 * @return 
 */
uint32 writeOneUIntExtern (longInt number, char* output, const char** errString)
{
  longInt shiftNumber = number;
  uint32 segLen = 0;
  int i;

  //trying to figure out how many 7bits do we need for number 
  while(shiftNumber > 0)
    {
      segLen++;
      shiftNumber = shiftNumber >> 7;
    }
  if(number == 0)
    segLen = 1;

  //split number from the most insignificant part
  for(i = segLen - 1; i >= 0; i--)
    {
      output[i] = number & 0x7F;
      if(i != segLen - 1) output [i] |= 0x80;
      number >>= 7;
    }
  return segLen;
 
}
/**
 * @brief This function write one longInt type into a char array, and update the writing position for the next writing. 
 * @param number -The number to write
 * @param updatingBuffer -The buffer to write to, will update the starting position of the buffer after one sucessful writing. 
 * @param errString -The error string 
 * @return The number of bytes written
 */

uint32
writeOneUInt (longInt number, char** updatingBuffer, const char** errString)
{
  longInt shiftNumber = number;
  uint32 segLen = 0;
  char* output = *updatingBuffer;
  int i;

  //trying to figure out how many 7bits do we need for number 
  while(shiftNumber > 0)
    {
      segLen++;
      shiftNumber = shiftNumber >> 7;
    }
  if(number == 0)
    segLen = 1;

  //split number from the most insignificant part
  for(i = segLen - 1; i >= 0; i--)
    {
      output[i] = number & 0x7F;
      if(i != segLen - 1) output [i] |= 0x80;
      number >>= 7;
    }

  //	output += segLen;
  *updatingBuffer += segLen;
  return segLen;
}

/**
 * @brief This is the API to split an unsigned int into the format for writing into files.!! The user is resiponsible for 
 * freeing the returned buffer!! or, there will be memory leakage.
 *
 * @param intArray The input array
 * @param length The length of the input array
 * @param outputLen The length of the output in byets
 * @param errString Error string passing to the user
 */
char*
writeUInt (const longInt* intArray, size_t length, size_t* outputLen, const char** errString)
{
  char* buffer = (char*) malloc (10 * length);
  char** updatedBufferP = (char**) malloc (sizeof (char*));
  *updatedBufferP = buffer;
  uint32 written = 0;
  uint32 i;
  for(i = 0; i < length; i++)
    {
      written += writeOneUInt (intArray[i], updatedBufferP, errString);
    }
  buffer = (char*) realloc (buffer, written * 8);
  *outputLen = written;
  free (updatedBufferP);
  return buffer;
}

