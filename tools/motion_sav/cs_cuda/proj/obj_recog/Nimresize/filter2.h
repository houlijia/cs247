/*
 * filter2.h
 *
 * Code generation for function 'filter2'
 *
 */

#ifndef __FILTER2_H__
#define __FILTER2_H__

/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rt_nonfinite.h"
#include "rtwtypes.h"
#include "Nimresize_types.h"

/* Function Declarations */
extern void b_filter2(const double b_data[], const int b_size[1], const double
                      x[307200], double y[307200]);
extern void filter2(const double b_data[], const int b_size[2], const double x
                    [307200], double y[307200]);

#endif

/* End of code generation (filter2.h) */
