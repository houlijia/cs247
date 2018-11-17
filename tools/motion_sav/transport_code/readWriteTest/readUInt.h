/**
 * @file readUInt.h
 * @brief The file for basic reading of uint array
 */

#ifndef READ_UINT_H
#define READ_UINT_H


#include "common_ljw.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief This is the funtion for reading one uint from buffer
   * @param inputArray -The input buffer
   * @param start -The start position for reading the uint 
   * @param end  -The end position for reading the uint, including it. 
   * we do not need the number of bytes read when processing the uint. users can get it from (end-start+1). It's impossible that we can read only  part of the
   * 		  uint, because the end position ensures that the ending of the uint is also in this buffer. the upper call will
   * 		  take care of the problems.   
   * @param errString -the string for telling users the errors
   *
   */
  longInt readOneUInt (char* inputArray, int start, int end, const char** errString);


  /**@brief This is a wrapper of the @ref readOneUInt function
   * It search from the start, and return the end position of the first number, 
   * and call teh readOneUInt to read it
   * @param inputArray -The input reading array
   * @param start -The staring position of reading, or think it as the offset from inputArray
   * @param endP -Pass out the end byte position of the uint
   * @param errString -Error string
   * @return The longInt value read
   */
  longInt readOnlyOneUInt (const char* inputArray, int start, int* endP, const char** errString);


  /**
   * @brief If we want $required number of numbers(uint/real), Find the end position of last number in
   * the input array. The maxLen is there to prevent accessing the space outside the input array. This is used to find the starting position of the data part in one measurement block (end+1). And, we need to find 8 numbers.
   * Assume that we can always find the required numbers, required >0
   * @param start The start index, >=0
   * @param maxLen The end index of the input array, if the array is of size L, it should be L-1
   * @param errString The error string
   * @param required The amount of numbers that we need to find/skip
   * @return The end positon of the last number
   */
  uint32 findPos (const unsigned char* input, const uint32 start, const uint32 required, const uint32 maxLen, const char** errString);


  /**
   * @brief This function will find the end position of last positioin of the integer, and the number of intergers inside [start, start+maxBytes].  
   * @param input The input array
   * @param start The start position to read
   * @param maxBytes The maxmium bytes we can parse
   * @param numberOfInt The pointer for passing out the numberOfInt inside the [start, start+maxBytes-1]
   * @param errString The error string 
   * @return The end position of the last integer.
   */
  uint32 findPosBasedOnBytes (const unsigned char* input, const uint32 start, const uint32 maxBytes, uint32* numberoOfInt, const char** errString);

/**
 * @brief Count how many integers are inside the buffer. 
 * @param input The input buffer
 * @param len The length of buffer
 * @return The number of integers inside the buffer 
 */
  uint32 countNumbers(const unsigned char* input, const uint32 len);
  
#ifdef __cplusplus
}
#endif


#endif
