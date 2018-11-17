/*
 * comp_writeUInt_mexutil.c
 *
 * Code generation for function 'comp_writeUInt_mexutil'
 *
 * C source code generated on: Thu Jun 05 15:39:00 2014
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "comp_writeUInt.h"
#include "comp_writeUInt_mexutil.h"

/* Function Definitions */
void error(const emlrtStack *sp, const mxArray *b, emlrtMCInfo *location)
{
  const mxArray *pArray;
  pArray = b;
  emlrtCallMATLABR2012b(sp, 0, NULL, 1, &pArray, "error", TRUE, location);
}

/* End of code generation (comp_writeUInt_mexutil.c) */
