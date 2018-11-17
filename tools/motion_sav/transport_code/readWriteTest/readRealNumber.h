/**
 * @file readRealNumber.h
 * @brief The file for basic reading of real number array
 */

#ifndef READ_REAL_NUMBER_H
#define READ_REAL_NUMBER_H


#include "common_ljw.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /**@brief This function reads a real number 
   * It search from the start, and return the end position of the first number, 
   * and call teh readOneRealNumber to read it
   * @param inputArray -The input reading array
   * @param bytesRead -The bytes have read 
   * @param errString -Error string
   */
  double readOnlyOneRealNumber (char* inputArray, uint32* bytesRead, const char** errString);

#ifdef __cplusplus
}
#endif


#endif
