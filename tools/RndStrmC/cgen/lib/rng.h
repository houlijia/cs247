/*
 * rng.h
 *
 * Code generation for function 'rng'
 *
 * C source code generated on: Wed Nov 13 14:08:38 2013
 *
 */

#ifndef __RNG_H__
#define __RNG_H__
/* Include files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "rtwtypes.h"
#include "get_rand_types.h"

/* Function Declarations */
extern void b_rng(uint32_T arg1);
extern void c_rng(uint32_T arg1_Method, const uint32_T arg1_State[625], const uint32_T arg1_LegacyRandnState[2]);
extern void rng(uint32_T *settings_Method, uint32_T *settings_Seed, uint32_T settings_State[625], uint32_T settings_LegacyRandnState[2]);
extern void seed_not_empty_init(void);
#endif
/* End of code generation (rng.h) */
