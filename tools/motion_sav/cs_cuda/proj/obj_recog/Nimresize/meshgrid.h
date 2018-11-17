/*
 * meshgrid.h
 *
 * Code generation for function 'meshgrid'
 *
 */

#ifndef __MESHGRID_H__
#define __MESHGRID_H__

/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rt_nonfinite.h"
#include "rtwtypes.h"
#include "Nimresize_types.h"

/* Function Declarations */
extern void meshgrid(const emxArray_real_T *x, const emxArray_real_T *y,
                     emxArray_real_T *xx, emxArray_real_T *yy);

#endif

/* End of code generation (meshgrid.h) */
