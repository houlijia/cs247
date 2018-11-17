/*
 * meshgrid.c
 *
 * Code generation for function 'meshgrid'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "meshgrid.h"
#include "Nimresize_emxutil.h"
#include <stdio.h>

/* Function Definitions */
void meshgrid(const emxArray_real_T *x, const emxArray_real_T *y,
              emxArray_real_T *xx, emxArray_real_T *yy)
{
  emxArray_real_T *a;
  int ibtile;
  int jcol;
  int varargin_1_idx_0;
  int k;
  int y_idx_0;
  emxInit_real_T(&a, 2);
  if ((x->size[1] == 0) || (y->size[1] == 0)) {
    ibtile = xx->size[0] * xx->size[1];
    xx->size[0] = 0;
    xx->size[1] = 0;
    emxEnsureCapacity((emxArray__common *)xx, ibtile, (int)sizeof(double));
    ibtile = yy->size[0] * yy->size[1];
    yy->size[0] = 0;
    yy->size[1] = 0;
    emxEnsureCapacity((emxArray__common *)yy, ibtile, (int)sizeof(double));
  } else {
    jcol = x->size[1];
    ibtile = a->size[0] * a->size[1];
    a->size[0] = 1;
    a->size[1] = jcol;
    emxEnsureCapacity((emxArray__common *)a, ibtile, (int)sizeof(double));
    for (ibtile = 0; ibtile < jcol; ibtile++) {
      a->data[a->size[0] * ibtile] = x->data[ibtile];
    }

    varargin_1_idx_0 = y->size[1];
    ibtile = xx->size[0] * xx->size[1];
    xx->size[0] = varargin_1_idx_0;
    xx->size[1] = a->size[1];
    emxEnsureCapacity((emxArray__common *)xx, ibtile, (int)sizeof(double));
    for (jcol = 0; jcol + 1 <= a->size[1]; jcol++) {
      ibtile = jcol * varargin_1_idx_0;
      for (k = 1; k <= varargin_1_idx_0; k++) {
        xx->data[(ibtile + k) - 1] = a->data[jcol];
      }
    }

    varargin_1_idx_0 = x->size[1];
    y_idx_0 = y->size[1];
    ibtile = yy->size[0] * yy->size[1];
    yy->size[0] = y_idx_0;
    yy->size[1] = varargin_1_idx_0;
    emxEnsureCapacity((emxArray__common *)yy, ibtile, (int)sizeof(double));
    y_idx_0 = y->size[1];
    for (jcol = 1; jcol <= varargin_1_idx_0; jcol++) {
      ibtile = (jcol - 1) * y_idx_0;
      for (k = 1; k <= y_idx_0; k++) {
        yy->data[(ibtile + k) - 1] = y->data[k - 1];
      }
    }
  }

  emxFree_real_T(&a);
}

/* End of code generation (meshgrid.c) */
