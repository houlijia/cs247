#ifndef CODE_ELEMENT_H
#define CODE_ELEMENT_H

#include <stdio.h>
#include "common_ljw.h"
#include "DropList.h"

#ifdef __cplusplus
extern "C"
{
#endif


  class CodeElement
  {
  public:

    CodeElement () : data (NULL), isAllLenSet (0){ };

    //we want a deep copy here, so can not use the default copy constructor
    CodeElement (const CodeElement& ce);

    ~CodeElement ()
    {
      if (data != NULL)
        {
          free (this->data);
          this->data = NULL;
        }
    }


    /**
     * @brief This is the fuction to write a CodeElement into file
     * @param fp -The file handler to write to
     * @param errString -Error string
     */
    void writeToFile (FILE* fp, const char** errString);

    /**
     * @brief If the user is sure that the buffer size is enough for writing, just call this function, or call the function below
     */

    uint32 writeToBuffer (char* dst, const char** errString);


    /**
     * @brief Write the codeElement to buffer. If it is too large, return by parameter the leftBytes. Or, leftBytes =0.
     * @param offset The offset of intergers
     * @return The number of bytes that have been written to dst
     */


    uint32 writeDataToBuffer (char* dst, uint32 dstCapacity, const uint8 QNo, uint32* offset, const longInt startPos, uint32* leftBytes, const char** errString);



uint32 writeDataToBufferDrop (char* dst, uint32 dstCapacity, const uint8 QNo, uint32* offset, const longInt startPos, uint32* leftBytesP,   const char** errString);

    /**
     * @brief This function make a deep copy of a code element
     * if the dst has a data part that is not NULL, we will first free that data.
     * @param src -Source of clone operation
     * @param dst -Destination of clone
     * @param errString -Error
     */
    friend void clone_CE (const CodeElement * const src, CodeElement * const dst, const char** errString);

    /**
     * @brief This function calculate the position of data part of one measurement block, and set it to the pointer 'realData'. The lenght of realData is set to realLength. It will also change the length of the CodeElement to the size of meta data. Then, we can use the write function of CodeElment to write the code element. And, deal with the data part seperately. 
     * @param metaDataSize -The size of the meta data in one measurement block. Currrently it is the size of 10 numbers, including len_s etc.
     */

    void
    calcRealDataPos (uint metaDataSize)
    {
      realData = this->data + metaDataSize;
      realLength = this->length - metaDataSize;
      this->setLength (metaDataSize);

    }

    /**
     * This function only read the key and length of one CE from buffer
     * @param buffer The buffer to read from  
     * @param errString
     * @return the byte length of the two values read 
     */

    uint32 readKeyLenFromBuffer (char* const buffer, const char** errString);




    /**
     * @brief Read one CodeElement from a buffer. This function is designed to be called for file reading.
     * though it can also be use to read multiple CE from a general buffer
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
    char readOneCEFromBuffer (char* bufferStart, uint32* bufferSizeP, char* bufferSizeChanged, char** bufferP, char* gotOne, int* left, const char** errString);

    const longInt
    getKey () const
    {
      return this->key;
    };
/**
 * This returns the length of the CE, for the measurements CE, it returns the length
 * at the beginning. But, after resetting the data and len at the receiver, the 
 * return value will be the length of block meta data. use getRealLength to get the 
 * real number of measurements bytes.
 * @return 
 */
    const longInt
    getLength () const
    {
      return this->length;
    };

    /**
     * Only counts the number of measurements bytes 
     */
    const longInt
    getRealLength () const
    {
      return realLength;
    };

    void
    setLength (longInt len0)
    {
      length = len0;
    };
/**
 * This returns a const data pointer. You can read from it, but can not write.  
 * @return 
 */
    const char*
    getData () const
    {
      return data;
    };
  const char*
    getRealData () const
    {
      return realData;
    };


    /**
     * @brief This function set the total length of the code element, including the size of key and length bytes
     */

    void
    setAllLen (longInt len0)
    {
      allLength = len0;
      isAllLenSet = true;
    }

    /**
     * @brief Set the data pointer in the CodeElement, and the allLength
     */
    void
    setData (char* const data0, const uint32 keyLenSize)
    {
      data = data0;
      setAllLen (keyLenSize + length);
    }

    longInt getAllLen (const char** errString);




  protected:
    longInt key;
    longInt length;
    longInt allLength; /**< The total length of the code element, including
                        * the lenght of key and length */
    char* data;
    bool isAllLenSet;  ///< a flag indicating whether the allLength field is set
                      ///< or not
    char* realData; ///< the start of the real data part, skipped the 8 numbers
    longInt realLength; //< the length of the real data part

  };


#ifdef __cplusplus
}
#endif

#endif
