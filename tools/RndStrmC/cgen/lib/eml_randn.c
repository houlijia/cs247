/*
 * eml_randn.c
 *
 * Code generation for function 'eml_randn'
 *
 * C source code generated on: Wed Nov 13 14:08:38 2013
 *
 */

/* Include files */
#include "get_rand.h"
#include "get_randi.h"
#include "get_randn.h"
#include "get_randperm.h"
#include "get_randperm1.h"
#include "eml_randn.h"
#include "get_rand_data.h"

/* Function Definitions */
void eml_randn_init(void)
{
  int32_T i;
  static const uint32_T uv8[2] = { 362436069U, 0U };

  b_method = 0U;
  for (i = 0; i < 2; i++) {
    state[i] = uv8[i];
  }

  state[1] = 521288629U;
}

/* End of code generation (eml_randn.c) */
