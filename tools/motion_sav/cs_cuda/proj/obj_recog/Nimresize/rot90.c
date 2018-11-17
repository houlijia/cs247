/*
 * rot90.c
 *
 * Code generation for function 'rot90'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "rot90.h"
#include <stdio.h>

/* Function Definitions */
void rot90(const double A_data[], const int A_size[2], double B_data[], int
           B_size[2])
{
  int n;
  int j;
  n = A_size[1];
  for (j = 0; j < 2; j++) {
    B_size[j] = A_size[j];
  }

  for (j = 1; j <= n; j++) {
    B_data[j - 1] = A_data[n - j];
  }
}

/* End of code generation (rot90.c) */
