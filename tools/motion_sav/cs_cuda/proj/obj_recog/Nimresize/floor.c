/*
 * floor.c
 *
 * Code generation for function 'floor'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "Nimresize.h"
#include "floor.h"
#include <stdio.h>

/* Function Definitions */
void b_floor(double x[2])
{
  int k;
  for (k = 0; k < 2; k++) {
    x[k] = floor(x[k]);
  }
}

/* End of code generation (floor.c) */
