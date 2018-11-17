/*
 * rand.h
 *
 * Code generation for function 'rand'
 *
 * C source code generated on: Wed Nov 13 14:08:39 2013
 *
 */

#ifndef __RAND_H__
#define __RAND_H__
/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "rtwtypes.h"
#include "get_rand_types.h"

/* Function Declarations */
extern void b_rand(uint32_T varargin_2, emxArray_real_T *r);
extern void c_rand(int32_T varargin_2, emxArray_real_T *r);
extern real_T d_rand(void);
extern void eml_rand_mt19937ar(uint32_T e_state[625]);
extern void genrand_uint32_vector(uint32_T mt[625], uint32_T u[2]);
extern real_T genrandu(uint32_T mt[625]);
#endif
/* End of code generation (rand.h) */
