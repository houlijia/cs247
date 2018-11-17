/*
 * get_rand_initialize.c
 *
 * Code generation for function 'get_rand_initialize'
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
#include "get_rand_initialize.h"
#include "eml_rand_shr3cong_stateful.h"
#include "eml_rand_mcg16807_stateful.h"
#include "eml_randn.h"
#include "eml_rand.h"
#include "rng.h"
#include "eml_rand_mt19937ar_stateful.h"

/* Function Definitions */
void get_rand_initialize(void)
{
  state_not_empty_init();
  seed_not_empty_init();
  eml_rand_init();
  eml_randn_init();
  eml_rand_mcg16807_stateful_init();
  eml_rand_shr3cong_stateful_init();
}

/* End of code generation (get_rand_initialize.c) */
