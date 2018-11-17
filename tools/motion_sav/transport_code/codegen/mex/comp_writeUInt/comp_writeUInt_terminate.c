/*
 * comp_writeUInt_terminate.c
 *
 * Code generation for function 'comp_writeUInt_terminate'
 *
 * C source code generated on: Thu Jun 05 15:39:01 2014
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "comp_writeUInt.h"
#include "comp_writeUInt_terminate.h"

/* Function Definitions */
void comp_writeUInt_atexit(emlrtStack *sp)
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  sp->tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(sp);
  emlrtLeaveRtStackR2012b(sp);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

void comp_writeUInt_terminate(emlrtStack *sp)
{
  emlrtLeaveRtStackR2012b(sp);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (comp_writeUInt_terminate.c) */
