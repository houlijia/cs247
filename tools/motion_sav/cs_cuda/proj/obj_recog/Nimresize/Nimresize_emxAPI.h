/*
 * Nimresize_emxAPI.h
 *
 * Code generation for function 'Nimresize_emxAPI'
 *
 */

#ifndef __NIMRESIZE_EMXAPI_H__
#define __NIMRESIZE_EMXAPI_H__

/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rt_nonfinite.h"
#include "rtwtypes.h"
#include "Nimresize_types.h"

/* Function Declarations */
#ifdef __cplusplus
extern "C" {
#endif

extern emxArray_uint8_T *emxCreateND_uint8_T(int numDimensions, int *size);
extern emxArray_uint8_T *emxCreateWrapperND_uint8_T(unsigned char *data, int
  numDimensions, int *size);
extern emxArray_uint8_T *emxCreateWrapper_uint8_T(unsigned char *data, int rows,
  int cols);
extern emxArray_uint8_T *emxCreate_uint8_T(int rows, int cols);
extern void emxDestroyArray_uint8_T(emxArray_uint8_T *emxArray);
extern void emxInitArray_uint8_T(emxArray_uint8_T **pEmxArray, int numDimensions);

#ifdef __cplusplus
}
#endif
#endif

/* End of code generation (Nimresize_emxAPI.h) */
