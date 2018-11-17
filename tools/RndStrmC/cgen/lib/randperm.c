/*
 * randperm.c
 *
 * Code generation for function 'randperm'
 *
 * C source code generated on: Wed Nov 13 14:08:39 2013
 *
 */

/* Include files */
#include "get_rand.h"
#include "get_randi.h"
#include "get_randn.h"
#include "get_randperm.h"
#include "get_randperm1.h"
#include "randperm.h"
#include "rand.h"
#include "get_rand_emxutil.h"

/* Function Definitions */

/*
 *
 */
void b_randperm(uint32_T n, emxArray_real_T *p)
{
  uint32_T uv7[2];
  int32_T i;
  emxArray_int32_T *idx;
  int32_T k;
  boolean_T b_p;
  emxArray_int32_T *idx0;
  int32_T i2;
  int32_T j;
  int32_T pEnd;
  int32_T c_p;
  int32_T q;
  int32_T qEnd;
  int32_T kEnd;
  emxArray_real_T *b_idx;
  int32_T d_p[2];
  b_rand(n, p);
  for (i = 0; i < 2; i++) {
    uv7[i] = (uint32_T)p->size[i];
  }

  emxInit_int32_T(&idx, 2);
  i = idx->size[0] * idx->size[1];
  idx->size[0] = 1;
  idx->size[1] = (int32_T)uv7[1];
  emxEnsureCapacity((emxArray__common *)idx, i, (int32_T)sizeof(int32_T));
  if (p->size[1] == 0) {
  } else {
    for (k = 1; k <= p->size[1]; k++) {
      idx->data[k - 1] = k;
    }

    for (k = 1; k <= p->size[1] - 1; k += 2) {
      b_p = (p->data[k - 1] <= p->data[k]);
      if (b_p) {
      } else {
        idx->data[k - 1] = k + 1;
        idx->data[k] = k;
      }
    }

    b_emxInit_int32_T(&idx0, 1);
    i = idx0->size[0];
    idx0->size[0] = p->size[1];
    emxEnsureCapacity((emxArray__common *)idx0, i, (int32_T)sizeof(int32_T));
    k = p->size[1];
    for (i = 0; i < k; i++) {
      idx0->data[i] = 1;
    }

    i = 2;
    while (i < p->size[1]) {
      i2 = i << 1;
      j = 1;
      for (pEnd = 1 + i; pEnd < p->size[1] + 1; pEnd = qEnd + i) {
        c_p = j;
        q = pEnd;
        qEnd = j + i2;
        if (qEnd > p->size[1] + 1) {
          qEnd = p->size[1] + 1;
        }

        k = 0;
        kEnd = qEnd - j;
        while (k + 1 <= kEnd) {
          b_p = (p->data[idx->data[c_p - 1] - 1] <= p->data[idx->data[q - 1] - 1]);
          if (b_p) {
            idx0->data[k] = idx->data[c_p - 1];
            c_p++;
            if (c_p == pEnd) {
              while (q < qEnd) {
                k++;
                idx0->data[k] = idx->data[q - 1];
                q++;
              }
            }
          } else {
            idx0->data[k] = idx->data[q - 1];
            q++;
            if (q == qEnd) {
              while (c_p < pEnd) {
                k++;
                idx0->data[k] = idx->data[c_p - 1];
                c_p++;
              }
            }
          }

          k++;
        }

        for (k = 0; k + 1 <= kEnd; k++) {
          idx->data[(j + k) - 1] = idx0->data[k];
        }

        j = qEnd;
      }

      i = i2;
    }

    emxFree_int32_T(&idx0);
  }

  emxInit_real_T(&b_idx, 1);
  k = idx->size[1];
  i = b_idx->size[0];
  b_idx->size[0] = k;
  emxEnsureCapacity((emxArray__common *)b_idx, i, (int32_T)sizeof(real_T));
  for (i = 0; i < k; i++) {
    b_idx->data[i] = idx->data[i];
  }

  emxFree_int32_T(&idx);
  for (i = 0; i < 2; i++) {
    d_p[i] = p->size[i];
  }

  i = p->size[0] * p->size[1];
  p->size[0] = 1;
  p->size[1] = d_p[1];
  emxEnsureCapacity((emxArray__common *)p, i, (int32_T)sizeof(real_T));
  k = d_p[1];
  for (i = 0; i < k; i++) {
    p->data[p->size[0] * i] = b_idx->data[d_p[0] * i];
  }

  emxFree_real_T(&b_idx);
}

/*
 *
 */
void randperm(uint32_T n, uint32_T k, emxArray_real_T *p)
{
  int32_T m;
  int32_T loop_ub;
  real_T selectedLoc;
  real_T j;
  real_T t;
  real_T denom;
  real_T nleftm1;
  real_T newEntry;
  emxArray_real_T *hashTbl;
  emxArray_real_T *link;
  emxArray_real_T *val;
  emxArray_real_T *loc;
  m = p->size[0] * p->size[1];
  p->size[0] = 1;
  p->size[1] = (int32_T)k;
  emxEnsureCapacity((emxArray__common *)p, m, (int32_T)sizeof(real_T));
  loop_ub = (int32_T)k;
  for (m = 0; m < loop_ub; m++) {
    p->data[m] = 0.0;
  }

  if (k == 0U) {
  } else if (k >= n) {
    p->data[0] = 1.0;
    for (m = 0; m < (int32_T)((real_T)n - 1.0); m++) {
      selectedLoc = d_rand();
      j = floor(selectedLoc * ((1.0 + (real_T)m) + 1.0));
      p->data[(int32_T)((1.0 + (real_T)m) + 1.0) - 1] = p->data[(int32_T)(j +
        1.0) - 1];
      p->data[(int32_T)(j + 1.0) - 1] = (1.0 + (real_T)m) + 1.0;
    }
  } else if (k >= (real_T)n / 4.0) {
    t = 0.0;
    for (m = 0; m < (int32_T)(((real_T)k - 1.0) + 1.0); m++) {
      selectedLoc = (real_T)k - (real_T)m;
      denom = (real_T)n - t;
      nleftm1 = selectedLoc / denom;
      newEntry = d_rand();
      while (newEntry > nleftm1) {
        t++;
        denom--;
        nleftm1 += (1.0 - nleftm1) * (selectedLoc / denom);
      }

      t++;
      selectedLoc = d_rand();
      j = floor(selectedLoc * ((real_T)m + 1.0));
      p->data[m] = p->data[(int32_T)(j + 1.0) - 1];
      p->data[(int32_T)(j + 1.0) - 1] = t;
    }
  } else {
    emxInit_real_T(&hashTbl, 1);
    m = hashTbl->size[0];
    hashTbl->size[0] = (int32_T)k;
    emxEnsureCapacity((emxArray__common *)hashTbl, m, (int32_T)sizeof(real_T));
    loop_ub = (int32_T)k;
    for (m = 0; m < loop_ub; m++) {
      hashTbl->data[m] = 0.0;
    }

    emxInit_real_T(&link, 1);
    m = link->size[0];
    link->size[0] = (int32_T)k;
    emxEnsureCapacity((emxArray__common *)link, m, (int32_T)sizeof(real_T));
    loop_ub = (int32_T)k;
    for (m = 0; m < loop_ub; m++) {
      link->data[m] = 0.0;
    }

    emxInit_real_T(&val, 1);
    m = val->size[0];
    val->size[0] = (int32_T)k;
    emxEnsureCapacity((emxArray__common *)val, m, (int32_T)sizeof(real_T));
    loop_ub = (int32_T)k;
    for (m = 0; m < loop_ub; m++) {
      val->data[m] = 0.0;
    }

    emxInit_real_T(&loc, 1);
    m = loc->size[0];
    loc->size[0] = (int32_T)k;
    emxEnsureCapacity((emxArray__common *)loc, m, (int32_T)sizeof(real_T));
    loop_ub = (int32_T)k;
    for (m = 0; m < loop_ub; m++) {
      loc->data[m] = 0.0;
    }

    newEntry = 1.0;
    for (m = 0; m < (int32_T)k; m++) {
      nleftm1 = (real_T)n - (1.0 + (real_T)m);
      selectedLoc = d_rand();
      selectedLoc = floor(selectedLoc * (nleftm1 + 1.0));
      denom = selectedLoc - floor(selectedLoc / (real_T)k) * (real_T)k;
      j = hashTbl->data[(int32_T)(1.0 + denom) - 1];
      while ((j > 0.0) && (loc->data[(int32_T)j - 1] != selectedLoc)) {
        j = link->data[(int32_T)j - 1];
      }

      if (j > 0.0) {
        p->data[m] = val->data[(int32_T)j - 1] + 1.0;
      } else {
        p->data[m] = selectedLoc + 1.0;
        j = newEntry;
        newEntry++;
        loc->data[(int32_T)j - 1] = selectedLoc;
        link->data[(int32_T)j - 1] = hashTbl->data[(int32_T)(1.0 + denom) - 1];
        hashTbl->data[(int32_T)(1.0 + denom) - 1] = j;
      }

      if (m + 1 < (int32_T)k) {
        denom = nleftm1 - floor(nleftm1 / (real_T)k) * (real_T)k;
        selectedLoc = hashTbl->data[(int32_T)(1.0 + denom) - 1];
        while ((selectedLoc > 0.0) && (loc->data[(int32_T)selectedLoc - 1] !=
                nleftm1)) {
          selectedLoc = link->data[(int32_T)selectedLoc - 1];
        }

        if (selectedLoc > 0.0) {
          selectedLoc = val->data[(int32_T)selectedLoc - 1];
        } else {
          selectedLoc = nleftm1;
        }

        val->data[(int32_T)j - 1] = selectedLoc;
      }
    }

    emxFree_real_T(&loc);
    emxFree_real_T(&val);
    emxFree_real_T(&link);
    emxFree_real_T(&hashTbl);
  }
}

/* End of code generation (randperm.c) */
