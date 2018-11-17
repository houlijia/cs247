#ifndef WRITEUINT_H
#define WRITEUINT_H

#include "common_ljw.h"


#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief This is the API to split an unsigned int into the format for writing into files.!! The user is resiponsible for 
   * freeing the returned buffer!! or, there will be memory leakage.
   *
   * @param intArray The input array
   * @param length The length of the input array
   * @param outputLen The length of the output
   * @param errString Error string passing to the user
   */
  extern char* writeUInt (const longInt* intArray, size_t length, size_t* outputLen, const char** errString);


extern uint32 writeOneUIntExtern (longInt number, char* writingBuffer, const char** errString);

#ifdef __cplusplus
}
#endif

#endif
