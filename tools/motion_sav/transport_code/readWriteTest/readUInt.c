
#include "common_ljw.h"
#include <stdio.h>
#include <assert.h>

longInt
readOneUInt (char* inputArray, int start, int end, const char** errString)
{
  longInt returnV = 0;
  uint32 i = 0;
  if(start < 0 || end < 0 || start > end)
    *errString = "index error in readOneUInt";
  while(start <= end)
    {
      returnV |= (inputArray[end] & 0x7F) << (i * 7);
      end--;
      i++;
    }
  return returnV;
}

uint32 countNumbers(const unsigned char* input, const uint32 len)
  {
   uint32 result = 0;
    uint32 i;
   for(i =0; i<len; i++)
    {
     if ((input[i] & 0x80) ==0)
      {
       result++;
      }
    }
   return result;
      
  }

/**@brief This is a wrapper of the @ref readOneUInt function
 * It search from the start, and return the end position of the first number, 
 * and call teh readOneUInt to read it
 * @param inputArray -The input reading array
 * @param start -The staring position of reading, or think it as the offset from inputArray
 * @param endP -Pass out the end byte position of the uint
 * @param errString -Error string
 */

longInt
readOnlyOneUInt (char* inputArray, int start, int* endP, const char** errString)
{

  int j;
  ///printf("##\n");	
  for(j = 0; j < 10; j++)
    {
      //printf("%c ", inputArray[start+j]);	
      if(((inputArray[start + j]) & 0x80) == 0)
        {
          *endP = start + j;
          break;
        }
    }
  //	printf("\n");
  if(j == 10)
    {
      *errString = "uint length not found error\n";
      return -1;
    }
  else
    return readOneUInt (inputArray, start, *endP, errString);
}

uint32
findPos (const unsigned char* input, const uint32 start, const uint32 required, const uint32 maxLen, const char** errString)
{
  assert (required > 0);
  uint32 current = start;

  //This is the position after the end, we used < instead of <=
  uint32 endN = start + maxLen;
  uint32 gotNumber = 0;
  while(current < endN)
    {
      if((input[current] & 0x80) == 0)
        {
          gotNumber++;
          if(gotNumber == required)
            break;
        }
      current++;
    }
  if(current == endN && gotNumber != required)
    {
      *errString = "find pos not found error\n";
      fputs (*errString, stderr);
      return -1;
    }

  return current;
}

uint32
findPosBasedOnBytes (const unsigned char* input, const uint32 start, const uint32 maxBytes, uint32* numberOfInt, const char** errString)
{
  uint32 current = start;
  uint32 gotNumber = 0;
  uint32 lastZeroPos = 0;
  while(current < start + maxBytes)
    {
      if((input[current] & 0x80) == 0)
        {
          gotNumber++;
          lastZeroPos = current;

        }
      current++;
    }
  *numberOfInt = gotNumber;
  
  assert (gotNumber >= 1);

  return lastZeroPos;
}






#if 0
//readFile

/**
 * @brief if the users wants to know how many numbers still needed to be read in the next call, simply use required - readLen, 
 *
 */


void
readUIntArray (const char* inputArray, /* Input byte array */
               uint32 inputLen,
               longInt* outputArray, const uint32 required, uint32* outputLen, uint32* readBytesNum,\
		const char** errString)
{
  uint32 i, p, iRead, available, nbytes, nbytesOne;
  bool isLeft = 0;
  uint32 leftUnread = inputLen;
  p = 0;
  iRead = 0;
  uint32 segN = inputLen / 7 + 1;
  uint32* startPosArray = (uint32*) malloc (segN * sizeof (uint32));
  if(startPosArray == NULL)
    {
      *errString = "startPosArray malloc failed";
      return;
    }
  memset (startPosArray, 0, sizeof (startPosArray));
  for(i = 0; i < inputLen; i++)
    {
      if(inputArray[i] & 0x80 == 0)
        {
          startPosArray[p] = i;
          p++;
        }
    }
  //if the last one's most significant bit is 1 or not, we need to return this to the caller. But acturally, the caller knows that, and should be able to handle that 
  //        available = ((inputArray[inputLen-1] & 0x80) !=0 ) ? p-1 : p;
  //	assert(required < available);
  nbytes = 0;
  while(iRead < required)
    {
      outputArray[iRead] = readOneUInt (inputArray, startPosArray[iRead], startPosArray[iRead + 1] - 1, &nbytesOne, errString);
      nbytes += nbytesOne;
      iRead++;
      if(iRead + 1 > p)
        break;
    }
  *readBytesNum = nbytes;
  *outputLen = iRead;

  free (startPosArray);

}
#endif
