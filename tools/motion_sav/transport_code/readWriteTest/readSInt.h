/**
 * @file readSInt.h
 * @brief The file for basic reading of signed int array
 */

#ifndef READ_SINT_H
#define READ_SINT_H


#include "common_ljw.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief This is the funtion for reading one signed int from buffer
   * @param inputArray -The input buffer
   * @param start -The start position for reading the uint 
   * @param end  -The end position for reading the uint, including it. 
   * we do not need the number of bytes read when processing the uint. users can get it from (end-start+1). It's impossible that we can read only  part of the
   * 		  uint, because the end position ensures that the ending of the uint is also in this buffer. the upper call will
   * 		  take care of the problems.   
   * @param errString -the string for telling users the errors
   *
   */
  SlongInt readOneSInt (char* inputArray, uint32 start, uint32 end, const char** errString);


  /**@brief This is a wrapper of the @ref readOneSInt function
   * It search from the start, and return the end position of the first number, 
   * and call teh readOneSInt to read it
   * @param inputArray -The input reading array
   * @param start -The staring position of reading, or think it as the offset from inputArray
   * @param endP -Pass out the end byte position of the uint
   * @param errString -Error string
   */
  SlongInt readOnlyOneSInt (char* inputArray, int start, int* endP, const char** errString);

#ifdef __cplusplus
}
#endif


#endif
