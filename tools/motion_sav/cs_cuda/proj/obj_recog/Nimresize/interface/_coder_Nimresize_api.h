/*
 * _coder_Nimresize_api.h
 *
 * Code generation for function '_coder_Nimresize_api'
 *
 */

#ifndef ___CODER_NIMRESIZE_API_H__
#define ___CODER_NIMRESIZE_API_H__

/* Include files */
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include <stddef.h>
#include <stdlib.h>
#include "_coder_Nimresize_api.h"

/* Type Definitions */
#ifndef struct_emxArray_uint8_T
#define struct_emxArray_uint8_T

struct emxArray_uint8_T
{
  uint8_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_uint8_T*/

#ifndef typedef_emxArray_uint8_T
#define typedef_emxArray_uint8_T

typedef struct emxArray_uint8_T emxArray_uint8_T;

#endif                                 /*typedef_emxArray_uint8_T*/

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void Nimresize(uint8_T A[921600], real_T m[2], emxArray_uint8_T *rout);
extern void Nimresize_api(const mxArray *prhs[2], const mxArray *plhs[1]);
extern void Nimresize_atexit(void);
extern void Nimresize_initialize(void);
extern void Nimresize_terminate(void);
extern void Nimresize_xil_terminate(void);

#endif

/* End of code generation (_coder_Nimresize_api.h) */
