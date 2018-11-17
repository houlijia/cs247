/*
 * comp_writeUInt_api.c
 *
 * Code generation for function 'comp_writeUInt_api'
 *
 * C source code generated on: Thu Jun 05 15:39:01 2014
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "comp_writeUInt.h"
#include "comp_writeUInt_api.h"
#include "comp_writeUInt_emxutil.h"

/* Variable Definitions */
static emlrtRTEInfo e_emlrtRTEI = { 1, 1, "comp_writeUInt_api", "" };

/* Function Declarations */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, emxArray_real_T *y);
static const mxArray *b_emlrt_marshallOut(real_T u);
static uint32_T c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *nv,
  const char_T *identifier);
static const mxArray *c_emlrt_marshallOut(const emlrtStack *sp, char_T u_data[27],
  int32_T u_size[2]);
static uint32_T d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId);
static void e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId, emxArray_real_T *ret);
static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *val, const
  char_T *identifier, emxArray_real_T *y);
static const mxArray *emlrt_marshallOut(emxArray_uint8_T *u);
static uint32_T f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId);

/* Function Definitions */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, emxArray_real_T *y)
{
  e_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static const mxArray *b_emlrt_marshallOut(real_T u)
{
  const mxArray *y;
  const mxArray *m4;
  y = NULL;
  m4 = mxCreateDoubleScalar(u);
  emlrtAssign(&y, m4);
  return y;
}

static uint32_T c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *nv,
  const char_T *identifier)
{
  uint32_T y;
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = identifier;
  thisId.fParent = NULL;
  y = d_emlrt_marshallIn(sp, emlrtAlias(nv), &thisId);
  emlrtDestroyArray(&nv);
  return y;
}

static const mxArray *c_emlrt_marshallOut(const emlrtStack *sp, char_T u_data[27],
  int32_T u_size[2])
{
  const mxArray *y;
  const mxArray *m5;
  y = NULL;
  m5 = mxCreateCharArray(2, u_size);
  emlrtInitCharArrayR2013a(sp, u_size[1], m5, (char_T *)u_data);
  emlrtAssign(&y, m5);
  return y;
}

static uint32_T d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId)
{
  uint32_T y;
  y = f_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static void e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId, emxArray_real_T *ret)
{
  int32_T iv5[1];
  boolean_T bv0[1];
  int32_T iv6[1];
  iv5[0] = -1;
  bv0[0] = TRUE;
  emlrtCheckVsBuiltInR2012b(sp, msgId, src, "double", FALSE, 1U, iv5, bv0, iv6);
  ret->size[0] = iv6[0];
  ret->allocatedSize = ret->size[0];
  ret->data = (real_T *)mxGetData(src);
  ret->canFreeData = FALSE;
  emlrtDestroyArray(&src);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *val, const
  char_T *identifier, emxArray_real_T *y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = identifier;
  thisId.fParent = NULL;
  b_emlrt_marshallIn(sp, emlrtAlias(val), &thisId, y);
  emlrtDestroyArray(&val);
}

static const mxArray *emlrt_marshallOut(emxArray_uint8_T *u)
{
  const mxArray *y;
  static const int32_T iv4[2] = { 0, 0 };

  const mxArray *m3;
  y = NULL;
  m3 = mxCreateNumericArray(2, (int32_T *)&iv4, mxUINT8_CLASS, mxREAL);
  mxSetData((mxArray *)m3, (void *)u->data);
  mxSetDimensions((mxArray *)m3, u->size, 2);
  emlrtAssign(&y, m3);
  return y;
}

static uint32_T f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId)
{
  uint32_T ret;
  emlrtCheckBuiltInR2012b(sp, msgId, src, "uint32", FALSE, 0U, 0);
  ret = *(uint32_T *)mxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

void comp_writeUInt_api(emlrtStack *sp, const mxArray *prhs[2], const mxArray
  *plhs[3])
{
  emxArray_real_T *val;
  emxArray_uint8_T *output;
  uint32_T nv;
  int32_T errString_size[2];
  char_T errString_data[27];
  real_T cnt;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_real_T(sp, &val, 1, &e_emlrtRTEI, TRUE);
  emxInit_uint8_T(sp, &output, 2, &e_emlrtRTEI, TRUE);
  prhs[0] = emlrtProtectR2012b(prhs[0], 0, FALSE, -1);

  /* Marshall function inputs */
  emlrt_marshallIn(sp, emlrtAlias(prhs[0]), "val", val);
  nv = c_emlrt_marshallIn(sp, emlrtAliasP(prhs[1]), "nv");

  /* Invoke the target function */
  comp_writeUInt(sp, val, nv, output, &cnt, errString_data, errString_size);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(output);
  plhs[1] = b_emlrt_marshallOut(cnt);
  plhs[2] = c_emlrt_marshallOut(sp, errString_data, errString_size);
  output->canFreeData = FALSE;
  emxFree_uint8_T(&output);
  val->canFreeData = FALSE;
  emxFree_real_T(&val);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

/* End of code generation (comp_writeUInt_api.c) */
