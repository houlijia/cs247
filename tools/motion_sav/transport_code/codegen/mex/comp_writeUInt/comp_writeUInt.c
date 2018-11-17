/*
 * comp_writeUInt.c
 *
 * Code generation for function 'comp_writeUInt'
 *
 * C source code generated on: Thu Jun 05 15:39:01 2014
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "comp_writeUInt.h"
#include "comp_writeUInt_emxutil.h"
#include "eml_int_forloop_overflow_check.h"
#include "comp_writeUInt_mexutil.h"
#include "comp_writeUInt_data.h"

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = { 35, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo b_emlrtRSI = { 29, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo c_emlrtRSI = { 28, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo d_emlrtRSI = { 27, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo e_emlrtRSI = { 20, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo f_emlrtRSI = { 19, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo g_emlrtRSI = { 15, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo h_emlrtRSI = { 6, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRSInfo i_emlrtRSI = { 41, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtRSInfo j_emlrtRSI = { 232, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtRSInfo k_emlrtRSI = { 230, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtRSInfo l_emlrtRSI = { 61, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtRSInfo q_emlrtRSI = { 9, "eml_int_forloop_overflow_check",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"
};

static emlrtRSInfo r_emlrtRSI = { 12, "eml_int_forloop_overflow_check",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_int_forloop_overflow_check.m"
};

static emlrtRSInfo s_emlrtRSI = { 12, "round",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elfun/round.m" };

static emlrtRSInfo t_emlrtRSI = { 35, "eml_i64relops",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_i64relops.m" };

static emlrtRSInfo u_emlrtRSI = { 119, "eml_i64relops",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/eml/eml_i64relops.m" };

static emlrtRSInfo v_emlrtRSI = { 13, "bitand",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/ops/bitand.m" };

static emlrtRSInfo y_emlrtRSI = { 14, "fliplr",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/fliplr.m" };

static emlrtRSInfo ab_emlrtRSI = { 15, "fliplr",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/fliplr.m" };

static emlrtRSInfo bb_emlrtRSI = { 16, "fliplr",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/fliplr.m" };

static emlrtMCInfo emlrtMCI = { 65, 1, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtMCInfo b_emlrtMCI = { 239, 9, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtRTEInfo emlrtRTEI = { 1, 37, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRTEInfo c_emlrtRTEI = { 25, 17, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtRTEInfo d_emlrtRTEI = { 1, 53, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtBCInfo emlrtBCI = { -1, -1, 34, 31, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo b_emlrtBCI = { -1, -1, 34, 21, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo c_emlrtBCI = { -1, -1, 35, 35, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo d_emlrtBCI = { -1, -1, 37, 21, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo e_emlrtBCI = { -1, -1, 40, 39, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtECInfo emlrtECI = { -1, 40, 17, "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m"
};

static emlrtBCInfo f_emlrtBCI = { -1, -1, 21, 22, "val", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo g_emlrtBCI = { -1, -1, 33, 21, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo h_emlrtBCI = { -1, -1, 40, 17, "output", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtBCInfo i_emlrtBCI = { -1, -1, 28, 21, "code", "comp_writeUInt",
  "C:/cygwin/home/jianwel/csCode/CS.060214.1541/transport_code/cgen_src/comp_writeUInt.m",
  0 };

static emlrtRSInfo db_emlrtRSI = { 239, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

static emlrtRSInfo eb_emlrtRSI = { 65, "find",
  "C:/Program Files/MATLAB/R2013b/toolbox/eml/lib/matlab/elmat/find.m" };

/* Function Definitions */
void comp_writeUInt(const emlrtStack *sp, emxArray_real_T *val, uint32_T nv,
                    emxArray_uint8_T *output, real_T *cnt, char_T
                    errString_data[27], int32_T errString_size[2])
{
  emxArray_boolean_T *x;
  int32_T i0;
  int32_T idx;
  int32_T k;
  const mxArray *y;
  const mxArray *m0;
  boolean_T overflow;
  int32_T ii;
  boolean_T exitg2;
  const mxArray *b_y;
  int32_T i1;
  static const char_T cv0[27] = { 'N', 'e', 'g', 'a', 't', 'i', 'v', 'e', ' ',
    'i', 'n', 'p', 'u', 't', ' ', 't', 'o', ' ', 'w', 'r', 'i', 't', 'e', 'U',
    'I', 'n', 't' };

  emxArray_uint64_T *b_val;
  real_T d0;
  uint64_T vl;
  uint32_T c_y;
  boolean_T b0;
  emxArray_uint8_T *code;
  emxArray_int32_T *r0;
  emxArray_uint8_T *b_x;
  uint32_T n;
  int32_T exitg1;
  boolean_T b_vl;
  uint64_T qY;
  uint32_T b_qY;
  uint8_T xtmp;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_boolean_T(sp, &x, 1, &emlrtRTEI, TRUE);

  /* COMP_WRITEUINT Summary of this function goes here */
  /*    Detailed explanation goes here */
  errString_size[0] = 1;
  errString_size[1] = 0;
  *cnt = 0.0;
  st.site = &h_emlrtRSI;
  i0 = x->size[0];
  x->size[0] = val->size[0];
  emxEnsureCapacity(&st, (emxArray__common *)x, i0, (int32_T)sizeof(boolean_T),
                    &emlrtRTEI);
  idx = val->size[0];
  for (i0 = 0; i0 < idx; i0++) {
    x->data[i0] = (val->data[i0] < 0.0);
  }

  b_st.site = &i_emlrtRSI;
  c_st.site = &l_emlrtRSI;
  k = muIntScalarMin_sint32(1, x->size[0]);
  if (k <= x->size[0]) {
  } else {
    y = NULL;
    m0 = mxCreateString("Assertion failed.");
    emlrtAssign(&y, m0);
    c_st.site = &eb_emlrtRSI;
    error(&c_st, y, &emlrtMCI);
  }

  idx = 0;
  c_st.site = &k_emlrtRSI;
  if (1 > x->size[0]) {
    overflow = FALSE;
  } else {
    overflow = (x->size[0] > 2147483646);
  }

  if (overflow) {
    d_st.site = &r_emlrtRSI;
    check_forloop_overflow_error(&d_st);
  }

  ii = 1;
  exitg2 = FALSE;
  while ((exitg2 == FALSE) && (ii <= x->size[0])) {
    if (x->data[ii - 1]) {
      c_st.site = &j_emlrtRSI;
      idx = 1;
      exitg2 = TRUE;
    } else {
      ii++;
    }
  }

  emxFree_boolean_T(&x);
  if (idx <= k) {
  } else {
    b_y = NULL;
    m0 = mxCreateString("Assertion failed.");
    emlrtAssign(&b_y, m0);
    c_st.site = &db_emlrtRSI;
    error(&c_st, b_y, &b_emlrtMCI);
  }

  if (k == 1) {
    if (idx == 0) {
      k = 0;
    }
  } else {
    if (1 > idx) {
      i1 = -1;
    } else {
      i1 = 0;
    }

    k = i1 + 1;
  }

  if (!(k == 0)) {
    errString_size[0] = 1;
    errString_size[1] = 27;
    for (i0 = 0; i0 < 27; i0++) {
      errString_data[i0] = cv0[i0];
    }

    i0 = output->size[0] * output->size[1];
    output->size[0] = 1;
    output->size[1] = 1;
    emxEnsureCapacity(sp, (emxArray__common *)output, i0, (int32_T)sizeof
                      (uint8_T), &emlrtRTEI);
    output->data[0] = 0;
  } else {
    st.site = &g_emlrtRSI;
    i0 = val->size[0];
    for (k = 0; k < i0; k++) {
      b_st.site = &s_emlrtRSI;
      val->data[k] = muDoubleScalarRound(val->data[k]);
    }

    emxInit_uint64_T(&st, &b_val, 1, &d_emlrtRTEI, TRUE);
    i0 = b_val->size[0];
    b_val->size[0] = val->size[0];
    emxEnsureCapacity(sp, (emxArray__common *)b_val, i0, (int32_T)sizeof
                      (uint64_T), &emlrtRTEI);
    idx = val->size[0];
    for (i0 = 0; i0 < idx; i0++) {
      d0 = muDoubleScalarRound(val->data[i0]);
      if (d0 < 1.8446744073709552E+19) {
        if (d0 >= 0.0) {
          vl = (uint64_T)d0;
        } else {
          vl = 0ULL;
        }
      } else if (d0 >= 1.8446744073709552E+19) {
        vl = MAX_uint64_T;
      } else {
        vl = 0ULL;
      }

      b_val->data[i0] = vl;
    }

    st.site = &f_emlrtRSI;
    if (nv > 536870911U) {
      c_y = MAX_uint32_T;
    } else {
      c_y = nv << 3;
    }

    i0 = output->size[0] * output->size[1];
    output->size[0] = 1;
    output->size[1] = (int32_T)c_y;
    emxEnsureCapacity(sp, (emxArray__common *)output, i0, (int32_T)sizeof
                      (uint8_T), &emlrtRTEI);
    idx = (int32_T)c_y;
    for (i0 = 0; i0 < idx; i0++) {
      output->data[i0] = 0;
    }

    st.site = &e_emlrtRSI;
    b_st.site = &q_emlrtRSI;
    if (1U > nv) {
      b0 = FALSE;
    } else {
      b0 = (nv > 4294967294U);
    }

    if (b0) {
      b_st.site = &r_emlrtRSI;
      b_check_forloop_overflow_error(&b_st);
    }

    c_y = 1U;
    emxInit_uint8_T(sp, &code, 2, &c_emlrtRTEI, TRUE);
    emxInit_int32_T(sp, &r0, 2, &emlrtRTEI, TRUE);
    emxInit_uint8_T(sp, &b_x, 2, &emlrtRTEI, TRUE);
    while (c_y <= nv) {
      i0 = b_val->size[0];
      ii = (int32_T)c_y;
      vl = b_val->data[emlrtDynamicBoundsCheckFastR2012b(ii, 1, i0, &f_emlrtBCI,
        sp) - 1];

      /*  First generate an array representing the vector in a */
      /*  little endian order, then reverse the array and write it */
      /*  out */
      i0 = code->size[0] * code->size[1];
      code->size[0] = 1;
      code->size[1] = (int32_T)nv;
      emxEnsureCapacity(sp, (emxArray__common *)code, i0, (int32_T)sizeof
                        (uint8_T), &emlrtRTEI);
      idx = (int32_T)nv;
      for (i0 = 0; i0 < idx; i0++) {
        code->data[i0] = 0;
      }

      /*  Initial allocation */
      n = 1U;
      do {
        exitg1 = 0;
        st.site = &d_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        c_st.site = &u_emlrtRSI;
        if (vl >= 4503599627370496ULL) {
          b_vl = TRUE;
        } else {
          b_vl = (128.0 <= vl);
        }

        if (b_vl) {
          st.site = &c_emlrtRSI;
          b_st.site = &v_emlrtRSI;
          i0 = code->size[1];
          ii = (int32_T)n;
          qY = 128ULL + (vl & 127ULL);
          if (qY < 128ULL) {
            qY = MAX_uint64_T;
          }

          if (qY > 255ULL) {
            qY = 255ULL;
          }

          code->data[emlrtDynamicBoundsCheckFastR2012b(ii, 1, i0, &i_emlrtBCI,
            sp) - 1] = (uint8_T)qY;
          st.site = &b_emlrtRSI;
          vl >>= 7;
          n++;
          emlrtBreakCheckFastR2012b(emlrtBreakCheckR2012bFlagVar, sp);
        } else {
          exitg1 = 1;
        }
      } while (exitg1 == 0);

      if (n > 1U) {
        i0 = code->size[1];
        ii = (int32_T)n;
        qY = 128ULL + vl;
        if (qY < 128ULL) {
          qY = MAX_uint64_T;
        }

        if (qY > 255ULL) {
          qY = 255ULL;
        }

        code->data[emlrtDynamicBoundsCheckFastR2012b(ii, 1, i0, &g_emlrtBCI, sp)
          - 1] = (uint8_T)qY;
        i0 = code->size[1];
        emlrtDynamicBoundsCheckFastR2012b(1, 1, i0, &b_emlrtBCI, sp);
        i0 = code->size[1];
        emlrtDynamicBoundsCheckFastR2012b(1, 1, i0, &emlrtBCI, sp);
        idx = code->data[0];
        b_qY = idx - 128U;
        if (b_qY > (uint32_T)idx) {
          b_qY = 0U;
        }

        i0 = (int32_T)b_qY;
        code->data[0] = (uint8_T)i0;
        i0 = code->size[1];
        emlrtDynamicBoundsCheckFastR2012b(1, 1, i0, &c_emlrtBCI, sp);
        i0 = code->size[1];
        ii = (int32_T)n;
        emlrtDynamicBoundsCheckFastR2012b(ii, 1, i0, &c_emlrtBCI, sp);
        st.site = &emlrtRSI;
        i0 = b_x->size[0] * b_x->size[1];
        b_x->size[0] = 1;
        b_x->size[1] = (int32_T)n;
        emxEnsureCapacity(&st, (emxArray__common *)b_x, i0, (int32_T)sizeof
                          (uint8_T), &emlrtRTEI);
        idx = (int32_T)n;
        for (i0 = 0; i0 < idx; i0++) {
          b_x->data[b_x->size[0] * i0] = code->data[i0];
        }

        i0 = code->size[0] * code->size[1];
        code->size[0] = 1;
        code->size[1] = b_x->size[1];
        emxEnsureCapacity(&st, (emxArray__common *)code, i0, (int32_T)sizeof
                          (uint8_T), &emlrtRTEI);
        idx = b_x->size[0] * b_x->size[1];
        for (i0 = 0; i0 < idx; i0++) {
          code->data[i0] = b_x->data[i0];
        }

        b_st.site = &y_emlrtRSI;
        i0 = b_x->size[1];
        i0 += (i0 < 0);
        if (i0 >= 0) {
          idx = (int32_T)((uint32_T)i0 >> 1);
        } else {
          idx = ~(int32_T)((uint32_T)~i0 >> 1);
        }

        b_st.site = &ab_emlrtRSI;
        c_st.site = &q_emlrtRSI;
        for (ii = 1; ii <= idx; ii++) {
          b_st.site = &bb_emlrtRSI;
          k = b_x->size[1] - ii;
          b_st.site = &bb_emlrtRSI;
          xtmp = code->data[code->size[0] * (ii - 1)];
          code->data[code->size[0] * (ii - 1)] = code->data[code->size[0] * k];
          code->data[code->size[0] * k] = xtmp;
        }
      } else {
        i0 = code->size[1];
        emlrtDynamicBoundsCheckFastR2012b(1, 1, i0, &d_emlrtBCI, sp);
        if (vl > 255ULL) {
          vl = 255ULL;
        }

        code->data[0] = (uint8_T)vl;
      }

      i0 = code->size[1];
      emlrtDynamicBoundsCheckFastR2012b(1, 1, i0, &e_emlrtBCI, sp);
      i0 = code->size[1];
      ii = (int32_T)n;
      emlrtDynamicBoundsCheckFastR2012b(ii, 1, i0, &e_emlrtBCI, sp);
      i0 = r0->size[0] * r0->size[1];
      r0->size[0] = 1;
      r0->size[1] = (int32_T)((real_T)n - 1.0) + 1;
      emxEnsureCapacity(sp, (emxArray__common *)r0, i0, (int32_T)sizeof(int32_T),
                        &emlrtRTEI);
      idx = (int32_T)((real_T)n - 1.0);
      for (i0 = 0; i0 <= idx; i0++) {
        ii = output->size[1];
        k = (int32_T)(*cnt + (1.0 + (real_T)i0));
        r0->data[r0->size[0] * i0] = emlrtDynamicBoundsCheckFastR2012b(k, 1, ii,
          &h_emlrtBCI, sp);
      }

      i0 = r0->size[1];
      ii = (int32_T)n;
      emlrtSizeEqCheck1DFastR2012b(i0, ii, &emlrtECI, sp);
      idx = (int32_T)n;
      for (i0 = 0; i0 < idx; i0++) {
        output->data[r0->data[r0->size[0] * i0] - 1] = code->data[i0];
      }

      *cnt += (real_T)n;
      c_y++;
      emlrtBreakCheckFastR2012b(emlrtBreakCheckR2012bFlagVar, sp);
    }

    emxFree_uint8_T(&b_x);
    emxFree_int32_T(&r0);
    emxFree_uint64_T(&b_val);
    emxFree_uint8_T(&code);

    /* err = obj.write(output(1:cnt)); */
  }

  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

/* End of code generation (comp_writeUInt.c) */
