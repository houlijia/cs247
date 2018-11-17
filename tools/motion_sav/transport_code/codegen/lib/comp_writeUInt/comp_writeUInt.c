/*
 * comp_writeUInt.c
 *
 * Code generation for function 'comp_writeUInt'
 *
 * C source code generated on: Thu Jun 05 15:37:38 2014
 *
 */

/* Include files */
#include "comp_writeUInt.h"
#include "comp_writeUInt_emxutil.h"

/* Function Declarations */
static real_T rt_roundd(real_T u);

/* Function Definitions */
static real_T rt_roundd(real_T u)
{
  real_T y;
  if (fabs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = floor(u + 0.5);
    } else if (u > -0.5) {
      y = 0.0;
    } else {
      y = ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

/*
 * function [output, cnt, errString] = comp_writeUInt( val, nv )
 */
void comp_writeUInt(emxArray_real_T *val, uint32_T nv, emxArray_uint8_T *output,
                    real_T *cnt, char_T errString_data[27], int32_T
                    errString_size[2])
{
  emxArray_boolean_T *x;
  int32_T ii;
  int32_T idx;
  int32_T k;
  boolean_T exitg2;
  int32_T i0;
  static const char_T cv0[27] = { 'N', 'e', 'g', 'a', 't', 'i', 'v', 'e', ' ',
    'i', 'n', 'p', 'u', 't', ' ', 't', 'o', ' ', 'w', 'r', 'i', 't', 'e', 'U',
    'I', 'n', 't' };

  emxArray_uint64_T *b_val;
  real_T d0;
  uint64_T vl;
  uint32_T y;
  emxArray_uint8_T *code;
  emxArray_int32_T *r0;
  emxArray_uint8_T *b_x;
  uint32_T n;
  int32_T exitg1;
  boolean_T b_vl;
  uint64_T qY;
  uint32_T b_qY;
  uint8_T xtmp;
  emxInit_boolean_T(&x, 1);

  /* COMP_WRITEUINT Summary of this function goes here */
  /*    Detailed explanation goes here */
  /* 'comp_writeUInt:4' errString =''; */
  errString_size[0] = 1;
  errString_size[1] = 0;

  /* 'comp_writeUInt:5' cnt = 0; */
  *cnt = 0.0;

  /* 'comp_writeUInt:6' if ~isempty(find(val<0,1)) */
  ii = x->size[0];
  x->size[0] = val->size[0];
  emxEnsureCapacity((emxArray__common *)x, ii, (int32_T)sizeof(boolean_T));
  idx = val->size[0];
  for (ii = 0; ii < idx; ii++) {
    x->data[ii] = (val->data[ii] < 0.0);
  }

  if (1 <= x->size[0]) {
    k = 1;
  } else {
    k = x->size[0];
  }

  idx = 0;
  ii = 1;
  exitg2 = FALSE;
  while ((exitg2 == FALSE) && (ii <= x->size[0])) {
    if (x->data[ii - 1]) {
      idx = 1;
      exitg2 = TRUE;
    } else {
      ii++;
    }
  }

  emxFree_boolean_T(&x);
  if (k == 1) {
    if (idx == 0) {
      k = 0;
    }
  } else {
    if (1 > idx) {
      i0 = -1;
    } else {
      i0 = 0;
    }

    k = i0 + 1;
  }

  if (!(k == 0)) {
    /* 'comp_writeUInt:7' errString = 'Negative input to writeUInt'; */
    errString_size[0] = 1;
    errString_size[1] = 27;
    for (ii = 0; ii < 27; ii++) {
      errString_data[ii] = cv0[ii];
    }

    /* 'comp_writeUInt:8' output=uint8(0); */
    ii = output->size[0] * output->size[1];
    output->size[0] = 1;
    output->size[1] = 1;
    emxEnsureCapacity((emxArray__common *)output, ii, (int32_T)sizeof(uint8_T));
    output->data[0] = 0;
  } else {
    /* 'comp_writeUInt:12' if islogical(val) */
    /* 'comp_writeUInt:14' elseif ~isinteger(val) */
    /* 'comp_writeUInt:15' val = round(val); */
    ii = val->size[0];
    for (k = 0; k < ii; k++) {
      val->data[k] = rt_roundd(val->data[k]);
    }

    emxInit_uint64_T(&b_val, 1);

    /* 'comp_writeUInt:17' val = uint64(val); */
    ii = b_val->size[0];
    b_val->size[0] = val->size[0];
    emxEnsureCapacity((emxArray__common *)b_val, ii, (int32_T)sizeof(uint64_T));
    idx = val->size[0];
    for (ii = 0; ii < idx; ii++) {
      d0 = rt_roundd(val->data[ii]);
      if (d0 < 1.8446744073709552E+19) {
        if (d0 >= 0.0) {
          vl = (uint64_T)d0;
        } else {
          vl = 0ULL;
        }
      } else {
        vl = MAX_uint64_T;
      }

      b_val->data[ii] = vl;
    }

    /* 'comp_writeUInt:19' output = uint8(zeros(1,8*nv)); */
    if (nv > 536870911U) {
      y = MAX_uint32_T;
    } else {
      y = nv << 3;
    }

    ii = output->size[0] * output->size[1];
    output->size[0] = 1;
    output->size[1] = (int32_T)y;
    emxEnsureCapacity((emxArray__common *)output, ii, (int32_T)sizeof(uint8_T));
    idx = (int32_T)y;
    for (ii = 0; ii < idx; ii++) {
      output->data[ii] = 0;
    }

    /* 'comp_writeUInt:20' for k=1:nv */
    y = 1U;
    emxInit_uint8_T(&code, 2);
    emxInit_int32_T(&r0, 2);
    emxInit_uint8_T(&b_x, 2);
    while (y <= nv) {
      /* 'comp_writeUInt:21' vl = val(k); */
      vl = b_val->data[(int32_T)y - 1];

      /*  First generate an array representing the vector in a */
      /*  little endian order, then reverse the array and write it */
      /*  out */
      /* 'comp_writeUInt:25' code = uint8(zeros(1,nv)); */
      ii = code->size[0] * code->size[1];
      code->size[0] = 1;
      code->size[1] = (int32_T)nv;
      emxEnsureCapacity((emxArray__common *)code, ii, (int32_T)sizeof(uint8_T));
      idx = (int32_T)nv;
      for (ii = 0; ii < idx; ii++) {
        code->data[ii] = 0;
      }

      /*  Initial allocation */
      /* 'comp_writeUInt:26' n = 1; */
      n = 1U;

      /* 'comp_writeUInt:27' while vl >=  128 */
      do {
        exitg1 = 0;
        if (vl >= 4503599627370496ULL) {
          b_vl = TRUE;
        } else {
          b_vl = (128.0 <= vl);
        }

        if (b_vl) {
          /* 'comp_writeUInt:28' code(n) = 128 + bitand(vl,127); */
          qY = 128ULL + (vl & 127ULL);
          if (qY < 128ULL) {
            qY = MAX_uint64_T;
          }

          if (qY > 255ULL) {
            qY = 255ULL;
          }

          code->data[(int32_T)n - 1] = (uint8_T)qY;

          /* 'comp_writeUInt:29' vl = bitshift(vl,-7); */
          vl >>= 7;

          /* 'comp_writeUInt:30' n = n+1; */
          n++;
        } else {
          exitg1 = 1;
        }
      } while (exitg1 == 0);

      /* 'comp_writeUInt:32' if n > 1 */
      if (n > 1U) {
        /* 'comp_writeUInt:33' code(n) = 128 + vl; */
        qY = 128ULL + vl;
        if (qY < 128ULL) {
          qY = MAX_uint64_T;
        }

        if (qY > 255ULL) {
          qY = 255ULL;
        }

        code->data[(int32_T)n - 1] = (uint8_T)qY;

        /* 'comp_writeUInt:34' code(1) = code(1) - 128; */
        idx = code->data[0];
        b_qY = idx - 128U;
        if (b_qY > (uint32_T)idx) {
          b_qY = 0U;
        }

        ii = (int32_T)b_qY;
        code->data[0] = (uint8_T)ii;

        /* 'comp_writeUInt:35' code = fliplr(code(1:n)); */
        ii = b_x->size[0] * b_x->size[1];
        b_x->size[0] = 1;
        b_x->size[1] = (int32_T)n;
        emxEnsureCapacity((emxArray__common *)b_x, ii, (int32_T)sizeof(uint8_T));
        idx = (int32_T)n;
        for (ii = 0; ii < idx; ii++) {
          b_x->data[b_x->size[0] * ii] = code->data[ii];
        }

        ii = code->size[0] * code->size[1];
        code->size[0] = 1;
        code->size[1] = b_x->size[1];
        emxEnsureCapacity((emxArray__common *)code, ii, (int32_T)sizeof(uint8_T));
        idx = b_x->size[0] * b_x->size[1];
        for (ii = 0; ii < idx; ii++) {
          code->data[ii] = b_x->data[ii];
        }

        ii = b_x->size[1];
        idx = ii / 2;
        for (ii = 1; ii <= idx; ii++) {
          k = b_x->size[1] - ii;
          xtmp = code->data[code->size[0] * (ii - 1)];
          code->data[code->size[0] * (ii - 1)] = code->data[code->size[0] * k];
          code->data[code->size[0] * k] = xtmp;
        }
      } else {
        /* 'comp_writeUInt:36' else */
        /* 'comp_writeUInt:37' code(1) = vl; */
        if (vl > 255ULL) {
          vl = 255ULL;
        }

        code->data[0] = (uint8_T)vl;
      }

      /* 'comp_writeUInt:40' output(cnt+1:cnt+n) = code(1:n); */
      ii = r0->size[0] * r0->size[1];
      r0->size[0] = 1;
      r0->size[1] = (int32_T)((real_T)n - 1.0) + 1;
      emxEnsureCapacity((emxArray__common *)r0, ii, (int32_T)sizeof(int32_T));
      idx = (int32_T)((real_T)n - 1.0);
      for (ii = 0; ii <= idx; ii++) {
        r0->data[r0->size[0] * ii] = (int32_T)(*cnt + (1.0 + (real_T)ii));
      }

      idx = (int32_T)n;
      for (ii = 0; ii < idx; ii++) {
        output->data[r0->data[r0->size[0] * ii] - 1] = code->data[ii];
      }

      /* 'comp_writeUInt:41' cnt = cnt + n; */
      *cnt += (real_T)n;
      y++;
    }

    emxFree_uint8_T(&b_x);
    emxFree_int32_T(&r0);
    emxFree_uint64_T(&b_val);
    emxFree_uint8_T(&code);

    /* err = obj.write(output(1:cnt)); */
  }
}

/* End of code generation (comp_writeUInt.c) */
