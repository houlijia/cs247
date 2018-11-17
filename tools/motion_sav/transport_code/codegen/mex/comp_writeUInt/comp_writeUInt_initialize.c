/*
 * comp_writeUInt_initialize.c
 *
 * Code generation for function 'comp_writeUInt_initialize'
 *
 * C source code generated on: Thu Jun 05 15:39:00 2014
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "comp_writeUInt.h"
#include "comp_writeUInt_initialize.h"
#include "comp_writeUInt_data.h"

/* Function Definitions */
void comp_writeUInt_initialize(emlrtStack *sp, emlrtContext *aContext)
{
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2012b();
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, aContext, NULL, 1);
  sp->tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(sp, FALSE, 0U, 0);
  emlrtEnterRtStackR2012b(sp);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

/* End of code generation (comp_writeUInt_initialize.c) */
