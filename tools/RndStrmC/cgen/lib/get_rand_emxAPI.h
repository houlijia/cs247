/*
 * get_rand_emxAPI.h
 *
 * Code generation for function 'get_rand_emxAPI'
 *
 * C source code generated on: Wed Nov 13 14:08:39 2013
 *
 */

#ifndef __GET_RAND_EMXAPI_H__
#define __GET_RAND_EMXAPI_H__
/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "rtwtypes.h"
#include "get_rand_types.h"

/* Function Declarations */
#ifdef __cplusplus
extern "C" {
#endif

extern emxArray_real_T *emxCreateND_real_T(int32_T numDimensions, int32_T *size);
extern emxArray_real_T *emxCreateWrapperND_real_T(real_T *data, int32_T numDimensions, int32_T *size);
extern emxArray_real_T *emxCreateWrapper_real_T(real_T *data, int32_T rows, int32_T cols);
extern emxArray_real_T *emxCreate_real_T(int32_T rows, int32_T cols);
extern void emxDestroyArray_real_T(emxArray_real_T *emxArray);

#ifdef __cplusplus
}
#endif

#endif
/* End of code generation (get_rand_emxAPI.h) */
