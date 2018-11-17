/*
 * eml_int_forloop_overflow_check.c
 *
 * Code generation for function 'eml_int_forloop_overflow_check'
 *
 * C source code generated on: Thu Jun 05 15:39:01 2014
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "comp_writeUInt.h"
#include "eml_int_forloop_overflow_check.h"
#include "comp_writeUInt_mexutil.h"

/* Variable Definitions */
static emlrtMCInfo c_emlrtMCI = { 52, 9, "eml_int_forloop_overflow_check",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"
};

static emlrtMCInfo d_emlrtMCI = { 51, 15, "eml_int_forloop_overflow_check",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"
};

static emlrtRSInfo cb_emlrtRSI = { 51, "eml_int_forloop_overflow_check",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"
};

static emlrtRSInfo fb_emlrtRSI = { 52, "eml_int_forloop_overflow_check",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"
};

/* Function Declarations */
static const mxArray *message(const emlrtStack *sp, const mxArray *b, const
  mxArray *c, emlrtMCInfo *location);

/* Function Definitions */
static const mxArray *message(const emlrtStack *sp, const mxArray *b, const
  mxArray *c, emlrtMCInfo *location)
{
  const mxArray *pArrays[2];
  const mxArray *m6;
  pArrays[0] = b;
  pArrays[1] = c;
  return emlrtCallMATLABR2012b(sp, 1, &m6, 2, pArrays, "message", TRUE, location);
}

void b_check_forloop_overflow_error(const emlrtStack *sp)
{
  const mxArray *y;
  static const int32_T iv2[2] = { 1, 34 };

  const mxArray *m2;
  char_T cv5[34];
  int32_T i;
  static const char_T cv6[34] = { 'C', 'o', 'd', 'e', 'r', ':', 't', 'o', 'o',
    'l', 'b', 'o', 'x', ':', 'i', 'n', 't', '_', 'f', 'o', 'r', 'l', 'o', 'o',
    'p', '_', 'o', 'v', 'e', 'r', 'f', 'l', 'o', 'w' };

  const mxArray *b_y;
  static const int32_T iv3[2] = { 1, 6 };

  char_T cv7[6];
  static const char_T cv8[6] = { 'u', 'i', 'n', 't', '3', '2' };

  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = sp;
  b_st.tls = sp->tls;
  y = NULL;
  m2 = mxCreateCharArray(2, iv2);
  for (i = 0; i < 34; i++) {
    cv5[i] = cv6[i];
  }

  emlrtInitCharArrayR2013a(sp, 34, m2, cv5);
  emlrtAssign(&y, m2);
  b_y = NULL;
  m2 = mxCreateCharArray(2, iv3);
  for (i = 0; i < 6; i++) {
    cv7[i] = cv8[i];
  }

  emlrtInitCharArrayR2013a(sp, 6, m2, cv7);
  emlrtAssign(&b_y, m2);
  st.site = &cb_emlrtRSI;
  b_st.site = &fb_emlrtRSI;
  error(&st, message(&b_st, y, b_y, &c_emlrtMCI), &d_emlrtMCI);
}

void check_forloop_overflow_error(const emlrtStack *sp)
{
  const mxArray *y;
  static const int32_T iv0[2] = { 1, 34 };

  const mxArray *m1;
  char_T cv1[34];
  int32_T i;
  static const char_T cv2[34] = { 'C', 'o', 'd', 'e', 'r', ':', 't', 'o', 'o',
    'l', 'b', 'o', 'x', ':', 'i', 'n', 't', '_', 'f', 'o', 'r', 'l', 'o', 'o',
    'p', '_', 'o', 'v', 'e', 'r', 'f', 'l', 'o', 'w' };

  const mxArray *b_y;
  static const int32_T iv1[2] = { 1, 23 };

  char_T cv3[23];
  static const char_T cv4[23] = { 'c', 'o', 'd', 'e', 'r', '.', 'i', 'n', 't',
    'e', 'r', 'n', 'a', 'l', '.', 'i', 'n', 'd', 'e', 'x', 'I', 'n', 't' };

  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = sp;
  b_st.tls = sp->tls;
  y = NULL;
  m1 = mxCreateCharArray(2, iv0);
  for (i = 0; i < 34; i++) {
    cv1[i] = cv2[i];
  }

  emlrtInitCharArrayR2013a(sp, 34, m1, cv1);
  emlrtAssign(&y, m1);
  b_y = NULL;
  m1 = mxCreateCharArray(2, iv1);
  for (i = 0; i < 23; i++) {
    cv3[i] = cv4[i];
  }

  emlrtInitCharArrayR2013a(sp, 23, m1, cv3);
  emlrtAssign(&b_y, m1);
  st.site = &cb_emlrtRSI;
  b_st.site = &fb_emlrtRSI;
  error(&st, message(&b_st, y, b_y, &c_emlrtMCI), &d_emlrtMCI);
}

/* End of code generation (eml_int_forloop_overflow_check.c) */
