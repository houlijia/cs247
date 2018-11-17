/*
 * comp_writeUInt_emxutil.h
 *
 * Code generation for function 'comp_writeUInt_emxutil'
 *
 * C source code generated on: Thu Jun 05 15:37:39 2014
 *
 */

#ifndef __COMP_WRITEUINT_EMXUTIL_H__
#define __COMP_WRITEUINT_EMXUTIL_H__
/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "rtwtypes.h"
#include "comp_writeUInt_types.h"

/* Function Declarations */
extern void emxEnsureCapacity(emxArray__common *emxArray, int32_T oldNumel, int32_T elementSize);
extern void emxFree_boolean_T(emxArray_boolean_T **pEmxArray);
extern void emxFree_int32_T(emxArray_int32_T **pEmxArray);
extern void emxFree_real_T(emxArray_real_T **pEmxArray);
extern void emxFree_uint64_T(emxArray_uint64_T **pEmxArray);
extern void emxFree_uint8_T(emxArray_uint8_T **pEmxArray);
extern void emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int32_T numDimensions);
extern void emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T numDimensions);
extern void emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
extern void emxInit_uint64_T(emxArray_uint64_T **pEmxArray, int32_T numDimensions);
extern void emxInit_uint8_T(emxArray_uint8_T **pEmxArray, int32_T numDimensions);
#endif
/* End of code generation (comp_writeUInt_emxutil.h) */
