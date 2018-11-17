/*
 * rdivide.c
 *
 * Code generation for function 'rdivide'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "rdivide.h"
#include <stdio.h>

/* Function Definitions */
void rdivide(const double x[2], const double y[2], double z[2])
{
  int i3;
  for (i3 = 0; i3 < 2; i3++) {
    z[i3] = x[i3] / y[i3];
  }
}

/* End of code generation (rdivide.c) */
