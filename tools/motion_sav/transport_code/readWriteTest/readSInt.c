
#include "common_ljw.h"

SlongInt
readOneSInt (char* inputArray, int start, int end, const char** errString)
{
  SlongInt returnV = 0;
  uint32 i = 0;
  if(start < 0 || end < 0 || start > end)
    *errString = "index error in readOneSInt";
  while(start <= end)
    {
      if(start != end)
        returnV |= (inputArray[end] & 0x7F) << (i * 7);
      else
        {

          returnV |= (inputArray[end] & 0x3F) << (i * 7);
          char isNeg = inputArray[start] & 0x40;
          if(isNeg)
            returnV *= -1;
        }

      end--;
      i++;
    }
  return returnV;
}

/**@brief This is a wrapper of the @ref readOneSInt function
 * It search from the start, and return the end position of the first number, 
 * and call teh readOneSInt to read it
 * @param inputArray -The input reading array
 * @param start -The staring position of reading, or think it as the offset from inputArray
 * @param endP -Pass out the end byte position of the uint
 * @param errString -Error string
 */

SlongInt
readOnlyOneSInt (char* inputArray, int start, int* endP, const char** errString)
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
      return;
    }
  else
    return readOneSInt (inputArray, start, *endP, errString);
}


#if 0
//readFile

/**
 * @brief if the users wants to know how many numbers still needed to be read in the next call, simply use required - readLen, 
 *
 */


void
readSIntArray (const char* inputArray, /* Input byte array */
               uint32 inputLen,
               SlongInt* outputArray, const uint32 required, uint32* outputLen, uint32* readBytesNum,\
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
      outputArray[iRead] = readOneSInt (inputArray, startPosArray[iRead], startPosArray[iRead + 1] - 1, &nbytesOne, errString);
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
