/*
 * eml_rand_shr3cong_stateful.c
 *
 * Code generation for function 'eml_rand_shr3cong_stateful'
 *
 * C source code generated on: Wed Nov 13 14:08:39 2013
 *
 */

/* Include files */
#include "get_rand.h"
#include "get_randi.h"
#include "get_randn.h"
#include "get_randperm.h"
#include "get_randperm1.h"
#include "eml_rand_shr3cong_stateful.h"
#include "get_rand_data.h"

/* Function Definitions */
void eml_rand_shr3cong_stateful_init(void)
{
  int32_T i1;
  for (i1 = 0; i1 < 2; i1++) {
    d_state[i1] = 362436069U + 158852560U * i1;
  }
}

/* End of code generation (eml_rand_shr3cong_stateful.c) */
