/*
 * filter2.c
 *
 * Code generation for function 'filter2'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "filter2.h"
#include "conv2.h"
#include "rot90.h"
#include <stdio.h>

/* Function Definitions */
void b_filter2(const double b_data[], const int b_size[1], const double x[307200],
               double y[307200])
{
  int m;
  double stencil_data[11];
  int stencil_size[1];
  int i;
  m = b_size[0];
  stencil_size[0] = b_size[0];
  for (i = 1; i <= m; i++) {
    stencil_data[i - 1] = b_data[m - i];
  }

  if (b_size[0] == 1) {
    conv2(stencil_data, stencil_size, x, y);
  } else {
    b_conv2(stencil_data, stencil_size, x, y);
  }
}

void filter2(const double b_data[], const int b_size[2], const double x[307200],
             double y[307200])
{
  int stencil_size[2];
  double stencil_data[11];
  int k;
  int j;
  int joffset;
  static double work[307200];
  int i;
  int ko;
  int jhi;
  int jmkom1;
  rot90(b_data, b_size, stencil_data, stencil_size);
  k = stencil_size[1] - 1;
  j = stencil_size[1] - 1;
  joffset = ((((k + (k < 0)) >> 1) + stencil_size[1]) - (((j + (j < 0)) >> 1) <<
              1)) - 2;
  for (k = 0; k < 307200; k++) {
    work[k] = 0.0;
    y[k] = 0.0;
  }

  for (j = 0; j < 640; j++) {
    for (i = 0; i < 480; i++) {
      work[i + 480 * j] += x[i + 480 * j];
    }
  }

  for (k = 0; k + 1 <= stencil_size[1]; k++) {
    ko = k - joffset;
    jhi = ko + 639;
    if (ko + 639 > 640) {
      jhi = 640;
    }

    if (stencil_data[k] != 0.0) {
      if (ko > 0) {
        j = ko - 1;
      } else {
        j = 0;
      }

      while (j + 1 <= jhi) {
        jmkom1 = (j - ko) + 1;
        for (i = 0; i < 480; i++) {
          y[i + 480 * j] += work[i + 480 * jmkom1] * stencil_data[k];
        }

        j++;
      }
    }
  }
}

/* End of code generation (filter2.c) */
