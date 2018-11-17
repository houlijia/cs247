/*
 * conv2.c
 *
 * Code generation for function 'conv2'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "conv2.h"
#include <stdio.h>

/* Function Definitions */
void b_conv2(const double arg1_data[], const int arg1_size[1], const double
             arg3[307200], double c[307200])
{
  int j;
  int k;
  int ioffset;
  static double work[307200];
  int ko;
  int ilo;
  int ihi;
  int i;
  j = arg1_size[0] - 1;
  k = arg1_size[0] - 1;
  ioffset = ((((j + (j < 0)) >> 1) + arg1_size[0]) - (((k + (k < 0)) >> 1) << 1))
    - 2;
  for (j = 0; j < 307200; j++) {
    work[j] = 0.0;
    c[j] = 0.0;
  }

  for (k = 0; k + 1 <= arg1_size[0]; k++) {
    ko = k - ioffset;
    if (ko > 0) {
      ilo = ko;
    } else {
      ilo = 1;
    }

    ihi = ko + 479;
    if (ko + 479 > 480) {
      ihi = 480;
    }

    if (arg1_data[k] != 0.0) {
      for (j = 0; j < 640; j++) {
        for (i = ilo; i <= ihi; i++) {
          work[(i + 480 * j) - 1] += arg3[(i - ko) + 480 * j] * arg1_data[k];
        }
      }
    }
  }

  for (j = 0; j < 640; j++) {
    for (i = 0; i < 480; i++) {
      c[i + 480 * j] += work[i + 480 * j];
    }
  }
}

void conv2(const double arg2_data[], const int arg2_size[1], const double arg3
           [307200], double c[307200])
{
  int k;
  int j;
  int joffset;
  static double work[307200];
  int i;
  int ko;
  int jhi;
  int jmkom1;
  k = arg2_size[0] - 1;
  j = arg2_size[0] - 1;
  joffset = ((((k + (k < 0)) >> 1) + arg2_size[0]) - (((j + (j < 0)) >> 1) << 1))
    - 2;
  for (k = 0; k < 307200; k++) {
    work[k] = 0.0;
    c[k] = 0.0;
  }

  for (j = 0; j < 640; j++) {
    for (i = 0; i < 480; i++) {
      work[i + 480 * j] += arg3[i + 480 * j];
    }
  }

  for (k = 0; k + 1 <= arg2_size[0]; k++) {
    ko = k - joffset;
    jhi = ko + 639;
    if (ko + 639 > 640) {
      jhi = 640;
    }

    if (arg2_data[k] != 0.0) {
      if (ko > 0) {
        j = ko - 1;
      } else {
        j = 0;
      }

      while (j + 1 <= jhi) {
        jmkom1 = (j - ko) + 1;
        for (i = 0; i < 480; i++) {
          c[i + 480 * j] += work[i + 480 * jmkom1] * arg2_data[k];
        }

        j++;
      }
    }
  }
}

/* End of code generation (conv2.c) */
