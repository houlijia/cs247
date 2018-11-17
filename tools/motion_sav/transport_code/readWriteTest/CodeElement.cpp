
#include "writeUInt.h"
#include "CodeElement.h"
#include "readUInt.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <iostream>
using std::endl;
using std::cerr;

///@brief we want a deep copy here, so can not use the default copy constructor
CodeElement::CodeElement (const CodeElement& ce) : isAllLenSet (0)
{
  this->key = ce.key;
  this->length = ce.length;
  this->data = (char*) malloc (ce.length);
  memcpy (this->data, ce.data, this->length);
}


longInt
CodeElement::getAllLen (const char** errString)
{
  if(isAllLenSet)
    return allLength;
  else
    {
      *errString = "get all len error\n";
      fputs (*errString, stderr);
      exit (0);
    }
}

uint32
CodeElement::readKeyLenFromBuffer (char* const buffer, const char** errString)
{
  int end;
  this->key = readOnlyOneUInt (buffer, 0, &end, errString);
  this->length = readOnlyOneUInt (buffer, end + 1, &end, errString);
  return end + 1;
}

/**
 * @brief Read one CodeElement from a buffer. This function is designed to be called for file reading.
 * though it can also be used to read multiple CE from a general buffer
 * @param bufferStart -The start position of the the buffer that will read from 
 * @param bufferSizeP -The pointer to an integer indicating the size of the current buffer. (Since the buffer size may grow, if it is too 
 * small for reading a large CodeElement)
 * @param bufferSizeChanged -Flag indicating whether buffer size is changed during the function call
 * @param left -If only part of the CodeElement is at the end of the buffer, we will use this left argument to pass the number of left bytes to the 
 * caller of this function. The caller can then decide how to deal with it.   
 * @param bufferP -The pointer to read from, and will update it at the end of the function. 
 * 		  It will become the position that the next function call will read from
 * @param gotOne -Flag indicating whether one CodeElment is read successfully
 * @param errString - Error String.
 *
 */
char
CodeElement::readOneCEFromBuffer (char* bufferStart, uint32* bufferSizeP, char* bufferSizeChanged, char** bufferP, char* gotOne, int* left, const char** errString)
{
  char* currentP = *bufferP;
  //char* bufferStart = *bufferStartP;
  //	uint32 nbytes = 0;
  uint32 bufferSize = *bufferSizeP;
  short i, j;
  //longInt  len =0;
  i = 0;
  *gotOne = FALSE;

  //first find the end positions of key and length
  short endPos[2] = {-1, -1};
  for(j = 0; j < 20; j++)
    {
      //if the buffer ends in the middle of the key/length
      //tell users it's not finished, and the number of bytes left 
      //simplify the read functions, do not need to handle with 
      //situations like buffer ends while reading in the middle of key/length 
      if((currentP - bufferStart) + j + 1 > bufferSize)
        {
          *gotOne = FALSE;
          *left = bufferSize - (currentP - bufferStart);
          return TRUE;
        }
      if(((currentP[j]) & 0x80) == 0)
        {
          endPos[i] = j;
          i++;
        }
      //we want to get out as soon as we got the two positions
      if(i > 1) break;
    }
  if(j == 20)
    {
      *errString = "uint key and length position not found in readOneCEBuffer";
      fputs (*errString, stderr);
      return TRUE;

    }


  //only read the length first, will read the key if everything is all right
  this->length = readOneUInt (currentP, endPos[0] + 1, endPos[1], errString);
  checkError (*errString);


  //the len of header is (endPos[1] +1),
  //if we have moved it to the beginning of the buffer, and it is still not enough
  //we resize the buffer, and return to the main while loop to process
  if(bufferStart == currentP && length > bufferSize - endPos[1] - 1)
    {

      bufferStart = (char*) realloc (bufferStart, length + 32);
      if(bufferStart == NULL)
        {
          *errString = "memory realloc fail in readOneCEBuffer";
          fputs (*errString, stderr);
          return TRUE;
        }
      *bufferP = bufferStart;

      *bufferSizeChanged = TRUE;
      *bufferSizeP = length + 32;
      //	printf("changed buffer size to %d\n", *bufferSizeP);
      return FALSE;
    }


  //if the data is complete in this buffer
  uint32 allLen = (currentP - bufferStart) + 1 + endPos[1]+ (length);
  if(allLen <= bufferSize)
    {
      this->key = readOneUInt (currentP, 0, endPos[0], errString);
      checkError (*errString);
      this->data = (char*) malloc (length);
      memcpy (this->data, currentP + endPos[1] + 1, length);
      this->setAllLen (1 + endPos[1]+ (length));

      //really move the pointer if we can successfully read one 
      *bufferP = currentP + endPos[1] + length + 1;
      *gotOne = TRUE;
      if(allLen < bufferSize)
        return FALSE;
      else
        {
          *bufferP = bufferStart;
          return TRUE;
        }
    }
  else
    {
      *gotOne = FALSE;
      *left = bufferSize - (currentP - bufferStart);
      //if(*left <0) 
      //printf("false bufferSize = %d, current = %p, head = %p  left= %d\n", bufferSize, currentP, bufferStart, *left); 
      return TRUE;
    }

}

/**
 * @brief This is the fuction to write a CodeElement into file
 * @param fp -The file handler to write to
 * @param errString -Error string
 */
void
CodeElement::writeToFile (FILE* fp, const char** errString)
{
  longInt keyLen[2];
  keyLen[0] = this->key;
  keyLen[1] = this->length;
  char* keyLenBuffer;
  size_t outputLen;
  keyLenBuffer = writeUInt (keyLen, 2, &outputLen, errString);
  fwrite (keyLenBuffer, 1, outputLen, fp);
  fwrite (this->data, 1, this->length, fp);
  free (keyLenBuffer);
};

uint32
CodeElement::writeToBuffer (char* dst, const char** errString)
{
  //need to write key and length to rtp packet too
  longInt keyLen[2];
  keyLen[0] = this->key;
  keyLen[1] = this->length;
  char* keyLenBuffer;
  size_t outputLen;
  keyLenBuffer = writeUInt (keyLen, 2, &outputLen, errString);
  memcpy (dst, keyLenBuffer, outputLen);
  memcpy (dst + outputLen, this->data, this->length);
  free (keyLenBuffer);
  return outputLen + (this->length);

}

uint32
CodeElement::writeDataToBufferDrop (char* dst, uint32 dstCapacity, const uint8 QNo, uint32* offset, const longInt startPos, uint32* leftBytesP,   const char** errString)
{
  char* buffer;
  uint32 written = 0;
  size_t outputLen = 0;
  uint32 leftBytes = *leftBytesP;
  uint32 nInt = 0;
  const longInt cOffset = *offset;

  int endP;

    const char* dataStart = this->getData();
    int nbins = readOnlyOneUInt(dataStart , 0, &endP, errString);
    int fillValue = nbins +2;


  char* currentP = (this->realData) +(this->realLength) - (leftBytes);

  //	buffer = writeUInt(&QNo, 1, &outputLen, errString);
  //	memcpy(dst, &QNo, 1);
  //	written +=1;

  buffer = writeUInt (&cOffset, 1, &outputLen, errString);
  memcpy (dst + written, buffer, outputLen);
  written += outputLen;
  free (buffer);

  dstCapacity -= written;

  //write
  uint32 maxBytesLen = (dstCapacity < leftBytes) ? dstCapacity : leftBytes;

  uint32 nextEndPos = findPosBasedOnBytes ((unsigned char*) currentP, 0, maxBytesLen, &nInt, errString);
  uint32 copyLen = nextEndPos + 1;
  //cerr<<"nInt ="<<nInt<<"fillValue"<<fillValue<<endl;
  //memcpy (dst + written, currentP, copyLen);
  for(int i=0; i< nInt; i++)
       {
        written+= writeOneUIntExtern (fillValue, dst+written, errString);
       }
  //memset(dst + written,  0, nInt);
  //written += nInt;
  *leftBytesP -= copyLen;
  *offset += nInt;

  return written;
}

uint32
CodeElement::writeDataToBuffer (char* dst, uint32 dstCapacity, const uint8 QNo, uint32* offset, const longInt startPos, uint32* leftBytesP, const char** errString)
{
  char* buffer;
  uint32 written = 0;
  size_t outputLen = 0;
  uint32 leftBytes = *leftBytesP;
  uint32 nInt = 0;
  const longInt cOffset = *offset;
  char* currentP = (this->realData) +(this->realLength) - (leftBytes);

  //	buffer = writeUInt(&QNo, 1, &outputLen, errString);
  //	memcpy(dst, &QNo, 1);
  //	written +=1;

  buffer = writeUInt (&cOffset, 1, &outputLen, errString);
  memcpy (dst + written, buffer, outputLen);
  written += outputLen;
  free (buffer);

  dstCapacity -= written;

  //write
  uint32 maxBytesLen = (dstCapacity < leftBytes) ? dstCapacity : leftBytes;

  uint32 nextEndPos = findPosBasedOnBytes ((unsigned char*) currentP, 0, maxBytesLen, &nInt, errString);
  //if(QNo ==0)
   //printf("###################\nQNo= %d   \n", QNo);
   //printf("nInt = %d   ", nInt);
  uint32 copyLen = nextEndPos + 1;

  memcpy (dst + written, currentP, copyLen);

  written += copyLen;
  *leftBytesP -= copyLen;
  *offset += nInt;

  return written;
}

void
clone_CE (const CodeElement * const src, CodeElement * const dst, const char** errString)
{
  if(dst == NULL || src == NULL)
    {
      *errString = "clone src or dst is NULL\n";
      fputs(*errString, stderr);
      return;
    }

  if((dst->data) != NULL)
    {
      free (dst->data);
      dst->data = NULL;
    }
  dst->data = (char*) malloc (src->length);
  if(dst->data == NULL)
    {
      *errString = "malloc error in clone of CodeElement\n";
      fputs(*errString, stderr);
      return;
    }

  memcpy (dst->data, src->data, src->length);
  dst->length = src->length;
  dst->key = src->key;
  dst->allLength = src->allLength;
  dst->isAllLenSet = src->isAllLenSet;

}

//#ifdef __cplusplus
//}
//#endif
